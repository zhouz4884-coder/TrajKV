"""Online all-support residual selection with separately charged metadata reads."""
from collections import defaultdict
from pathlib import Path
import math,sys,time
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/experiments'))
from bank import ProgressiveBank,unique_bytes
from fused_progressive import decode_last_generic
from check_fused_progressive import unpack_planes
from fused_masked import attention

def deployed_values(s,ref,bits,rk=True,rv=True):
    bk=decode_last_generic(s.k_code,s.k_scale,s.k_min,128,bits).transpose(2,3).contiguous()
    bv=decode_last_generic(s.v_code,s.v_scale,s.v_min,128,bits)
    if ref is None:return bk,bv,bk,bv
    b,h,n,d=bk.shape;dk,dv=bk,bv
    if rk:
        c=unpack_planes(ref[0],2,(b,h,d,n)).reshape(b,h,d,-1,128)
        r=(c*ref[1].float()[...,None]+ref[2].float()[...,None]).reshape(b,h,d,n).transpose(2,3)
        dk=(bk.float()+r).half()
    if rv:
        c=unpack_planes(ref[3],1,(b,h,n,d)).reshape(b,h,n,-1,128)
        r=((2*c-1)*ref[4].float()[...,None]).reshape(b,h,n,d)
        dv=(bv.float()+r).half()
    return bk,bv,dk,dv

class A2Bank(ProgressiveBank):
    settings={}
    def __init__(self,*args,**kwargs):
        torch.cuda.synchronize();start=time.perf_counter()
        super().__init__(*args,**kwargs)
        torch.cuda.synchronize();self.encoding_wall_ms=1000*(time.perf_counter()-start)
    def _initialize(self,*args,**kwargs):
        super()._initialize(*args,**kwargs)
        # Original visual positions are immutable and identical at every layer.
        # Share their backing rather than owning 28 duplicate index vectors.
        shared_visual_pos=self.states[0].visual_pos
        for state in self.states:state.visual_pos=shared_visual_pos
        z=dict(type(self).settings);self.selector=z.get('selector','none');self.fraction=z.get('fraction',.5)
        self.page=z.get('page',256);self.diagnostic=z.get('diagnostic',False)
        self.local_calibration=z.get('local_calibration',False);self.local_gains=[];self.local_seen=defaultdict(int)
        self.meta=[];self.error_stats=[];self.query_stats=[];self.selection_events=[];self.evictions=[]
        self.drop_turns=z.get('drop_turns',[]);self.drop_before=z.get('drop_before',5)
        self.compact_history=z.get('compact_history',False);self.turn_boundaries={};self._last_logical=None
        if self.compact_history:self.compact([])
        self.events=defaultdict(lambda:dict(calls=0,query_tokens=0,base_visual_bytes=0,residual_bytes=0,
            text_bytes=0,reference_bytes=0,selection_bytes=0,position_bytes=0,selected_head_pages=0,available_head_pages=0,selection_skipped_calls=0))
        for layer,(s,r) in enumerate(zip(self.states,self.refinements)):
            if not r or not z.get('store_selector_meta',True):self.meta.append(None);continue
            bk,bv,dk,dv=deployed_values(s,r,self.bits,self.refine_k,self.refine_v)
            b,h,n,d=bk.shape
            if n%self.page:raise ValueError('Unaligned A2 page')
            k=bk.reshape(b,h,-1,self.page,d);v=bv.reshape(b,h,-1,self.page,d)
            ek=(dk.float()-bk.float()).reshape(b,h,-1,self.page,d)
            ev=(dv.float()-bv.float()).reshape(b,h,-1,self.page,d)
            mean=v.float().mean(-2)
            self.meta.append((k.amin(-2).contiguous(),k.amax(-2).contiguous(),
                ek.norm(dim=-1).amax(-1).contiguous(),ev.norm(dim=-1).amax(-1).contiguous(),
                (mean-mean.mean(-2,keepdim=True)).norm(dim=-1).contiguous()))
            if self.diagnostic and self.references[layer] is not None:
                original=self.references[layer]
                for axis,base,full in zip(('k','v'),(bk,bv),original):
                    energy=(base.float()-full.float()).square().reshape(b,h,-1,self.page,d).sum((-2,-1))
                    for owner in range(h):
                        vals=energy[0,owner].tolist();total=sum(vals);np=len(vals);ordered=sorted(vals,reverse=True)
                        self.error_stats.append({'layer':layer,'axis':axis,'kv_head':owner,'pages':np,'page':self.page,
                          'page_energy':vals,'top_fraction':math.ceil(np/4)/np,'top_fraction_energy':sum(ordered[:math.ceil(np/4)])/max(total,1e-30),
                          'top3_share':sum(ordered[:3])/max(total,1e-30),'top3_uniform_share':min(3,np)/np})
            del bk,bv,dk,dv,k,v,ek,ev,mean
        equivalents=z.get('global_read_equivalents')
        self.extra_budget=(int(sum(unique_bytes(r or []) for r in self.refinements)*self.fraction*equivalents)
                           if equivalents is not None else None)

    def select(self,q,i,remaining=None):
        meta=self.meta[i];s=self.states[i];h=s.text_k.shape[1];np=s.visual_pos.numel()//self.page;cells=h*np
        ref=self.refinements[i];active=(ref[:3] if self.refine_k else [])+(ref[3:] if self.refine_v else [])
        total=unique_bytes(active);unit=total//cells
        cap=int(total*self.fraction)
        if remaining is not None:cap=min(cap,max(0,remaining))
        if self.selector=='none':return None,0,0,0,0
        if self.selector=='all':
            return (None,total,0,cells,0) if remaining is None or total<=remaining else (None,0,0,0,1)
        if self.selector in ('periodic','prefill_only'):
            decode_index=sum(e['calls'] for key,e in self.events.items() if key[0]==self.turn and key[1]=='decode')//len(self.states)
            active_forward=self.phase.endswith('prefill') or (self.selector=='periodic' and decode_index%4==0)
            if remaining is not None and total>remaining:active_forward=False
            return None,total if active_forward else 0,0,cells if active_forward else 0,0
        if self.selector=='layer_uniform':
            period=round(1/self.fraction)
            # 28 layers divides by 2/4: same per-forward aggregate allowance.
            active_layer=i%period==(self.turn+q.shape[-2])%period
            if remaining is not None and total>remaining:active_layer=False
            return None,total if active_layer else 0,0,cells if active_layer else 0,0
        # Metadata and query payload plus explicit live selection temporaries.
        # This is a logical tensor-payload account, not all physical kernel traffic.
        mask_bytes=cells
        static=self.selector in ('random','spread','time_uniform','thumb_random')
        estimate=cells*32 if static else unique_bytes(meta)+q.numel()*q.element_size()+q.numel()*4*3+cells*4*(q.shape[1]//h*q.shape[-2]+6)
        overhead=estimate+mask_bytes
        count=min(cells,max(0,(cap-overhead)//unit))
        if count==0:return None,0,0,0,1
        mask=torch.zeros((1,h,np),device=q.device,dtype=torch.uint8)
        seq=sum(e['calls'] for e in self.events.values())
        if self.selector in ('random','thumb_random'):
            # Device-only stateless pseudo-random ranking; no answer/FP16/future information.
            gen=torch.Generator(device=q.device).manual_seed(20261113+i*100003+seq)
            score=torch.rand(cells,device=q.device,generator=gen)
            if self.selector=='thumb_random':
                # Verified GotOcr2 processor appends the full-image thumbnail.
                # Prioritize its last 256 visual positions within the same cap.
                slots=torch.arange(cells,device=q.device)
                score=score+2*(slots%np>=max(0,np-256//self.page)).to(score.dtype)
        elif self.selector=='spread':
            slots=(torch.arange(count,device=q.device)*cells//count+(seq+i))%cells
            mask.view(-1)[slots]=1
            return mask,count*unit,overhead,count,0
        elif self.selector=='time_uniform':
            # Spend this read's fractional budget on head-pages in rotating order.
            slots=(torch.arange(count,device=q.device)+seq*count+i)%cells;mask.view(-1)[slots]=1
            return mask,count*unit,overhead,count,0
        else:
            low,high,ke,ve,contrast=meta
            b,hq,nq,d=q.shape;qp=q.float().reshape(b,h,-1,d)
            upper=(torch.einsum('bhqd,bhpd->bhqp',qp.clamp_min(0),high.float())+
                   torch.einsum('bhqd,bhpd->bhqp',qp.clamp_max(0),low.float()))/math.sqrt(d)
            weight=(upper.logsumexp(-2)-math.log(qp.shape[-2])).softmax(-1)
            if self.selector=='mass':score=weight
            elif self.selector=='uncertainty':score=weight*(qp.norm(dim=-1).mean(-1)[...,None]*ke/math.sqrt(d)+ve)
            elif self.selector=='risk':score=weight*(qp.norm(dim=-1).mean(-1)[...,None]*ke/math.sqrt(d)*(contrast+1e-3)+ve)
            elif self.selector=='tie':
                proximity=(upper-upper.amax(-1,keepdim=True)).exp().mean(-2)
                score=proximity*(qp.norm(dim=-1).mean(-1)[...,None]*ke/math.sqrt(d)+ve)
            else:raise ValueError('Unknown selector '+self.selector)
        chosen=score.flatten().topk(count).indices
        mask.view(-1)[chosen]=1
        return mask,count*unit,overhead,count,0

    def read(self,q,i):
        s=self.states[i];ref=self.refinements[i];mask=None;res=sel=cnt=skip=0
        if ref:
            remaining=None if self.extra_budget is None else self.extra_budget-self.extra_read_bytes
            query_for_selector=q[:,:,-1:] if type(self).settings.get('selector_query_last',False) else q
            mask,res,sel,cnt,skip=self.select(query_for_selector,i,remaining)
            if cnt==0:ref=None
        key=(self.turn,self.phase,self.pass_kind,i);e=self.events[key];e['calls']+=1;e['query_tokens']+=q.shape[-2]
        e['position_bytes']+=8*(s.visual_pos.numel()+s.text_length)
        e['base_visual_bytes']+=unique_bytes(self._visual_tensors(s));e['text_bytes']+=sum(t[:,:,:s.text_length].numel()*t.element_size() for t in (s.text_k,s.text_v))
        e['residual_bytes']+=res;e['selection_bytes']+=sel;e['selected_head_pages']+=cnt
        e['available_head_pages']+=s.text_k.shape[1]*(s.visual_pos.numel()//self.page)
        e['selection_skipped_calls']+=skip;self.extra_read_bytes+=res+sel
        out=attention(q,s,refinement=ref,bits=self.bits,refine_k=self.refine_k,refine_v=self.refine_v,refinement_mask=mask,page_size=self.page,
            visual_logit_gain=type(self).settings.get('visual_logit_gain',1.),text_logit_gain=type(self).settings.get('text_logit_gain',1.))
        if self.local_calibration and self.local_seen[i]<2 and self.references[i] is not None:
            from types import SimpleNamespace
            view=SimpleNamespace(visual_k=self.references[i][0],visual_v=self.references[i][1],visual_pos=s.visual_pos,
                text_k=s.text_k,text_v=s.text_v,text_pos=s.text_pos,text_length=s.text_length,length=s.length)
            target=attention(q,view)
            base=attention(q,s,bits=self.bits)
            kr=attention(q,s,refinement=self.refinements[i],bits=self.bits,refine_k=True,refine_v=False)
            vr=attention(q,s,refinement=self.refinements[i],bits=self.bits,refine_k=False,refine_v=True)
            def mse(t):return float((t.float()-target.float()).square().mean())
            self.local_gains.append({'layer':i,'turn':self.turn,'event_index':self.local_seen[i],
                'mse_base':mse(base),'mse_kr':mse(kr),'mse_vr':mse(vr),'mse_both':mse(out),
                'feature_scope':'offline same-query attention-output calibration; no future labels'})
            self.local_seen[i]+=1
        if self.diagnostic and i in (0,7,14,27) and len(self.query_stats)<16 and self.refinements[i]:
            bk,bv,dk,dv=deployed_values(s,self.refinements[i],self.bits,self.refine_k,self.refine_v)
            k=bk.repeat_interleave(q.shape[1]//bk.shape[1],1)
            scores=(q.float()@k.float().transpose(-1,-2))/math.sqrt(q.shape[-1])
            bound=self.meta[i][2].repeat_interleave(self.page,-1).repeat_interleave(q.shape[1]//bk.shape[1],1)
            u=q.float().norm(dim=-1)[...,None]*bound[:,:,None,:]/math.sqrt(q.shape[-1])
            winner=scores.argmax(-1,keepdim=True);lower=(scores-u).gather(-1,winner)
            upper=scores+u;upper.scatter_(-1,winner,-torch.inf)
            certificate=lower>upper.amax(-1,keepdim=True)
            delta=((q.float()@(dk.float()-bk.float()).repeat_interleave(q.shape[1]//bk.shape[1],1).transpose(-1,-2))/math.sqrt(q.shape[-1]))
            self.query_stats.append({'layer':i,'turn':self.turn,'nq':q.shape[-2],
                'top1_rank_certificate_fraction':float(certificate.float().mean()),'mean_score_bound':float(u.mean()),
                'mean_actual_score_delta':float(delta.abs().mean()),'scope':'attention rank only; not output or answer safety'})
        return out

    def compact(self,remove_ranges):
        before=unique_bytes([t for s in self.states for t in (s.text_k,s.text_v,s.text_pos)])
        counts=[]
        for s in self.states:
            pos=s.text_pos[:s.text_length];keep=torch.ones_like(pos,dtype=torch.bool)
            for lo,hi in remove_ranges:keep&=~((pos>=lo)&(pos<hi))
            idx=keep.nonzero().flatten();oldpos=pos.index_select(0,idx).clone();n=idx.numel()
            # Same allocator policy for all C arms: exact live size at turn boundaries,
            # then a common 32-token growth reserve during generation.
            s.text_k=s.text_k[:,:,:s.text_length].index_select(2,idx).contiguous()
            s.text_v=s.text_v[:,:,:s.text_length].index_select(2,idx).contiguous()
            s.text_pos=oldpos;s.text_length=n;s.reserve=32;counts.append(n)
        after=unique_bytes([t for s in self.states for t in (s.text_k,s.text_v,s.text_pos)])
        self.evictions.append({'removed_ranges':remove_ranges,'text_owner_before':before,'text_owner_after':after,
            'saved_owner_bytes':before-after,'retained_tokens':counts[0],'logical_length':self.states[0].length,
            'original_positions_preserved':True,'later_states_not_reencoded':True})

    def on_turn_begin(self,number,logical):
        if self._last_logical is None:self._last_logical=logical
        self._turn_start=logical
        if self.compact_history:
            ranges=[self.turn_boundaries[t] for t in self.drop_turns if t in self.turn_boundaries] if number==self.drop_before else []
            self.compact(ranges)
    def on_turn_end(self,number,logical):self.turn_boundaries[number]=(self._turn_start,logical)
    def owned(self):
        result=super().owned();meta=[t for m in getattr(self,'meta',[]) if m for t in m]
        result['selector_metadata']=unique_bytes(meta);result['total_unique']+=result['selector_metadata']
        result['visual_position']=unique_bytes([s.visual_pos for s in self.states])
        result['text_position_capacity']=unique_bytes([s.text_pos for s in self.states])
        result['visual_with_metadata_ratio']=(result['visual_base']+result['visual_residual']+result['selector_metadata'])/self.visual_full_bytes
        return result
    def result_metadata(self):
        return {'selector':self.selector,'residual_fraction_cap':self.fraction,'page_size':self.page,
            'selection_logical_read_bytes':sum(e['selection_bytes'] for e in self.events.values()),
            'actual_residual_logical_read_bytes':sum(e['residual_bytes'] for e in self.events.values()),
            'global_extra_budget_bytes':self.extra_budget,'global_extra_budget_ok':self.extra_budget is None or self.extra_read_bytes<=self.extra_budget,
            'read_budget_bytes':self.extra_budget,'read_budget_ok':self.extra_budget is None or self.extra_read_bytes<=self.extra_budget,
            'budget_scope':('full residual payload for selected phases/layers, under the declared global cap' if self.selector in ('all','periodic','prefill_only','layer_uniform') else
                            'per forward aggregate cap = fraction * full residual payload, selection included; page selectors additionally cap each layer'),
            'no_reference_or_future_online_features':not self.diagnostic and not self.local_calibration,'page_error_diagnostics':self.error_stats,
            'query_diagnostics':self.query_stats,'history_evictions':self.evictions,
            'local_axis_calibration':self.local_gains,
            'encoding_wall_ms':self.encoding_wall_ms,
            'visual_logit_gain':type(self).settings.get('visual_logit_gain',1.),'text_logit_gain':type(self).settings.get('text_logit_gain',1.),
            'backend':'a2_all_visual_support_scalar_page_residual_branch',
            'read_byte_scope':'logical residual/base/text payload plus predeclared metadata/query/selector scratch charge; not measured physical DRAM'}
