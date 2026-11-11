"""Minimal query/value LoRA, following Hu et al. (2021), arXiv:2106.09685.

Original implementation, not vendored PEFT code. Frozen linear map plus alpha/r BA;
A has Kaiming initialization and B starts at zero. Merge only for final inference.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=16,dropout=0.):
        super().__init__()
        if not isinstance(base,nn.Linear) or rank<1: raise ValueError('Expected Linear and positive rank')
        self.base=base;self.rank=rank;self.scaling=alpha/rank
        self.dropout=nn.Dropout(dropout)
        self.lora_A=nn.Parameter(base.weight.new_empty(rank,base.in_features))
        self.lora_B=nn.Parameter(base.weight.new_zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
        for p in base.parameters(): p.requires_grad=False

    def forward(self,x):
        return self.base(x)+F.linear(F.linear(self.dropout(x),self.lora_A),self.lora_B)*self.scaling

    def merged(self):
        with torch.no_grad(): self.base.weight.add_(self.lora_B@self.lora_A,alpha=self.scaling)
        return self.base


def attach(model,rank=8,alpha=16,dropout=0.):
    if any(isinstance(m,LoRALinear) for m in model.modules()): raise ValueError('LoRA already attached')
    model.method='full' # Original encoder dropout active, as in standard full-model train mode.
    for p in model.parameters(): p.requires_grad=False
    names=[]
    for i,layer in enumerate(model.roberta.encoder.layer):
        for name in ('query','value'):
            setattr(layer.attention.self,name,LoRALinear(getattr(layer.attention.self,name),rank,alpha,dropout))
            names.append(f'roberta.encoder.layer.{i}.attention.self.{name}')
    for p in model.classifier.parameters(): p.requires_grad=True
    return names


def merge(model):
    count=0
    for layer in model.roberta.encoder.layer:
        for name in ('query','value'):
            module=getattr(layer.attention.self,name)
            if isinstance(module,LoRALinear):
                setattr(layer.attention.self,name,module.merged());count+=1
    if not count: raise ValueError('No adapters to merge')
    return model
