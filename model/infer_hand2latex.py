import argparse
import json
import math
from typing import List, Tuple
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models

# Tokenizer wrapper
class Tokenizer:
	def __init__(self,token_to_id: dict):
		self.t2i = token_to_id
		self.i2t = {int(v):k for k,v in token_to_id.items()}
		self.sos = self.t2i.get('<sos>')
		self.eos = self.t2i.get('<eos>')
		self.pad = self.t2i.get('<pad>')
		if self.sos is None or self.eos is None:
			raise ValueError('Tokenizer must contain <sos> and <eos>')

	def encode(self, tokens: List[str]) -> List[int]:
		return [self.t2i.get(t, self.t2i.get('<unk>', 0)) for t in tokens]
	
	def decode(self, ids: List[int]) -> List[str]:
		return [self.i2t.get(int(i), '<unk>') for i in ids]

# Basic CNN encoder	
class CNNEncoder(nn.Module):
	def __init__(self, out_dim = 256, pretrained=False):
		super().__init__()
		self.features = nn.Sequential(
			backbone.conv1,
			backbone.bn1,
			backbone.relu,
			backbone.maxpool,
			backbone.layer1,
			backbone.layer2,
			backbone.layer3,
			backbone.layer4,
		)
		self.project = nn.Conv2d(512, out_dim, kerner_size=1)
	
	def forward(self, x):
		feat = self.features(x)
		feat = self.project(feat)
		B, D, Hf, Wf = feat.shape
		seq = feat.permute(0, 2, 3, 1).contiguous().view(B, Hf*Wf, D)
		return seq

# Positional encoder
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=10000):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x);
		T = x.size(1)
		return x + self.pe[:T].unsqueeze(0)

