"""Shared explicit backend selection for free and teacher-forced GPU runs."""
from masked_bank import A2Bank
from rate_bank import RateBank
from allocation_bank import AllocationBank
from calib_bank import CalibBank
from calib_post import CalibPostBank
from hybrid_bank import HybridBank
from hybrid_packed_bank import HybridPackedBank
from single_axis_bank import SingleAxisBank
from oracle_bank import OracleKBank
from geometry_bank import GeometryBank
from rotated_key_port import RotatedKeyPort
from query_moments_bank import QueryMomentsBank
from calib_packed_bank import CalibPackedBank
from direction_bank import DirectionBank
from affine_rate_bank import AffineRateBank
from cached_refinement_bank import CachedRefinementBank
from anchor_bank import AnchoredAffineBank,AnchoredDirectionBank,AnchoredFullBank
from learned_direction_bank import LearnedDirectionBank
from metadata_direction_bank import MetadataDirectionBank
def bank_class(method):
 choices={'rate':RateBank,'affine_rate':AffineRateBank,'cached_asym':CachedRefinementBank,'alloc1':AllocationBank,'calibpost':CalibPostBank,'hybrid':HybridBank,
          'hybrid_packed':HybridPackedBank,'axisdiag':SingleAxisBank,'oraclek':OracleKBank,'geometrydiag':GeometryBank,'rotk_port':RotatedKeyPort,'querymoments':QueryMomentsBank,'calib_packed':CalibPackedBank,'direction':DirectionBank}
 choices.update(anchor_affine=AnchoredAffineBank,anchor_direction=AnchoredDirectionBank,anchor_full=AnchoredFullBank)
 choices['learned_direction']=LearnedDirectionBank
 choices['direction_meta']=MetadataDirectionBank
 return choices.get(method,CalibBank if method.startswith('calib') else A2Bank)
