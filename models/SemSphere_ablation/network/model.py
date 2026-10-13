"""Ablation-only assembly from unchanged SemSphere components."""
import torch
from torch import nn
from collections import defaultdict
from models.SemSphere.network import model as core

class base_Model(core.SemSphere):
 def __init__(self, configs, device):
  del device
  nn.Module.__init__(self)
  self.disable_temporal=getattr(configs,'disable_temporal',False)
  self.disable_behavior=getattr(configs,'disable_behavior',False)
  self.disable_oc=getattr(configs,'disable_oc',False)
  self.disable_recon=getattr(configs,'disable_recon',False)
  self.profile_timing=getattr(configs,'profile_timing',False)
  self.timing=defaultdict(float)
  h=configs.d_model
  channels=configs.input_channels
  dropout=getattr(configs,'dropout',0.1)
  self.temporal=None if self.disable_temporal else core.TemporalEncoder(h,configs.tcn_layers)
  if self.disable_behavior:
   self.prompt_gen=self.text_encoder=self.prompt_embedding=self.relation=None
  else:
   self.prompt_gen=core.BehaviorPromptGenerator(configs.domain_text,configs.variable_desc)
   self.text_encoder=core.FrozenTextEncoder(configs.text_model,h,
    allow_hash_fallback=getattr(configs,'allow_hash_text_encoder',False),
    enable_cache=getattr(configs,'text_embedding_cache',True),
    cache_on_device=getattr(configs,'text_cache_on_device',True))
   self.prompt_embedding=core.PrecomputedPromptEmbedding(self.text_encoder,self.prompt_gen,channels)
   self.relation=core.RelationAwareAttention(h)
  self.fusion=None if self.disable_temporal or self.disable_behavior else core.CrossModalFusion(h,configs.attention_heads,dropout)
  self.reconstruction=None if self.disable_recon else core.ReconstructionDecoder(h,configs.window_size,dropout)
  p=getattr(configs,'project_dim',32)
  self.one_class_head=None if self.disable_oc else core.OneClassProjectionHead(channels,h,p,
   getattr(configs,'oc_hidden_dim',None),getattr(configs,'oc_pooling','flatten'))
  self.register_buffer('oc_center',torch.zeros(p))
  # Skip training-only center initialization when the entire OC branch is absent.
  self.register_buffer('oc_center_initialized',torch.tensor(self.disable_oc))

 def forward(self,x,x_context=None):
  if not (self.disable_temporal or self.disable_behavior or self.disable_oc or self.disable_recon):
   return super().forward(x,x_context)
  batch,_,channels=x.shape
  zt=None if self.disable_temporal else self.temporal(x.float())
  ids=None if self.disable_behavior else self.prompt_gen(x)
  zb=None if self.disable_behavior else self.relation(self.prompt_embedding(ids))
  if self.disable_temporal:
   zu=zb;zt=torch.zeros_like(zb);attn=zu.new_zeros(batch,channels,channels)
  elif self.disable_behavior:
   zu=zt;zb=torch.zeros_like(zt);attn=zu.new_zeros(batch,channels,channels)
  else: zu,attn=self.fusion(zt,zb)
  # Identity placeholder satisfies the shared trainer API when the decoder is
  # absent. Its diagnostic reconstruction error is zero and is never scored.
  recon=x.float() if self.disable_recon else self.reconstruction(zu)
  if self.disable_oc:
   q=zu.new_zeros(batch,self.oc_center.numel());distance=zu.new_zeros(batch)
  else:
   q=self.one_class_head(zu);distance=(q-self.oc_center).square().mean(dim=-1)
  return {'x_recon':recon,'q':q,'oc_distance':distance,'z_temporal':zt,
   'z_semantic':zb,'z_unified':zu,'cross_attention':attn,'prompt_ids':ids}
