"""Six controlled SemSphere ablations; enabled losses retain baseline weights."""
ABLATION_CASES = {
 '02_no_behavior': {'label':'w/o Behavior Semantic','disable_temporal':False,'disable_behavior':True,'disable_oc':False,'disable_align':True,'disable_oc_score':False},
 '03_no_recon': {'label':'w/o Recon','disable_temporal':False,'disable_behavior':False,'disable_oc':False,'disable_align':False,'disable_oc_score':False,'disable_recon':True},
 '04_no_oc': {'label':'w/o OC','disable_temporal':False,'disable_behavior':False,'disable_oc':True,'disable_align':False,'disable_oc_score':True},
 '06_full': {'label':'SemSphere (full)','disable_temporal':False,'disable_behavior':False,'disable_oc':False,'disable_align':False,'disable_oc_score':False},
 '07_no_behavior_recon': {'label':'w/o Behavior Semantic + Recon','disable_temporal':False,'disable_behavior':True,'disable_oc':False,'disable_align':True,'disable_oc_score':False,'disable_recon':True},
 '08_no_behavior_oc': {'label':'w/o Behavior Semantic + OC','disable_temporal':False,'disable_behavior':True,'disable_oc':True,'disable_align':True,'disable_oc_score':True},
}
def apply_ablation_case(config, name):
 case=ABLATION_CASES[name]
 config.ablation_case=name
 for field,value in case.items(): setattr(config,field,value)
 if case['disable_align']:
  config.lambda_align=0.0;config.use_infonce=False
 if case.get('disable_recon',False):
  config.lambda_recon=0.0;config.score_alpha=0.0
  # OC is the only score: unit weight also covers baselines with beta=0.
  config.score_beta=1.0
 if case['disable_oc']: config.lambda_oc=0.0
 if case['disable_oc_score']: config.score_beta=0.0
 return config
