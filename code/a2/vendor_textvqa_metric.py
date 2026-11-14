"""Pin official Meta MMF TextVQA evaluator and standalone answer normalization."""
import ast,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 urls={'processors':'https://raw.githubusercontent.com/facebookresearch/mmf/main/mmf/datasets/processors/processors.py',
  'evaluators':'https://raw.githubusercontent.com/facebookresearch/mmf/main/mmf/utils/m4c_evaluators.py',
  'license':'https://raw.githubusercontent.com/facebookresearch/mmf/main/LICENSE'}
 dest=ROOT/'third_party/textvqa_metric_source';dest.mkdir(exist_ok=False,parents=True);sources={};audit={}
 for name,url in urls.items():
  blob=urllib.request.urlopen(url,timeout=60).read();(dest/(name+'.txt')).write_bytes(blob);sources[name]=blob.decode('utf-8');audit[name]={'url':url,'sha256':hashlib.sha256(blob).hexdigest()}
 classes=[]
 for source,name in [('processors','EvalAIAnswerProcessor'),('evaluators','TextVQAAccuracyEvaluator')]:
  node=next(n for n in ast.parse(sources[source]).body if isinstance(n,ast.ClassDef) and n.name==name)
  node.decorator_list=[]
  if name=='EvalAIAnswerProcessor':node.bases=[]
  classes.append(ast.unparse(node))
 header='# Copyright (c) Facebook, Inc. and its affiliates.\n# BSD-3-Clause; original source and LICENSE in third_party/textvqa_metric_source.\n# Standalone port: registry/base dependency removed, metric logic unchanged.\nimport re\nimport numpy as np\n\n'
 metric=ROOT/'code/a2/textvqa_metric.py';metric.write_text(header+'\n\n'.join(classes)+'\n',encoding='utf-8')
 audit['port_sha256']=hashlib.sha256(metric.read_bytes()).hexdigest();audit['changes']='AST extraction of2 official classes; remove registry decorator and BaseProcessor dependency; metric logic retained'
 (ROOT/'audit/a2/textvqa_metric_source.json').write_text(json.dumps(audit,indent=2));print(json.dumps({'classes':len(classes),'metric_file':str(metric),'sha256':audit['port_sha256']}))
if __name__=='__main__':main()
