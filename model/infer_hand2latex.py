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

	def forward(self, x):
		T = x.size(1)
		return x + self.pe[:T].unsqueeze(0)
# Img to Latex
class Image2Latex(nn.Module):
	def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, pad_idx=0):
		super().__init__()
		self.encoder = CNNEncoder(out_dim=d_model)
		self.pos_enc = PositionalEncoding(d_model)
		self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
		decode_layer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
		self.output_proj = nn.Linear(d_model, vocab_size)
		self.d_model = d_model

	def encode_images(self, images):
		mem = self.encoder(images)  # [B, S, D]
		mem = self.pos_enc(mem)
		# transformer expects seq first
		return mem.permute(1, 0 , 2) # [S, B, D]

	def decode_step(self, tgt_ids, memory, tgt_mask=None):
		# tgt_ids shape [B, T}
		tgt = self.tok_emb(tgt_ids) * math.sqrt(self.d_model)
		tgt = self.pos_enc(tgt)
		tgt = tgt.permute(1, 0, 2) # [T, B, D]
		if tgt_mask is None:
			T = tgt.size(0)
			tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
		out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
		out = out.permute(1, 0, 2) # [B,T,D]
		logits = self.output_proj(out) # [B, T, V]
		return logits

# Beam search
def beam_search(model: Image2Latex, memory: torch.Tensor, tokanizer: Tokenizer, beam_size: int = 5, max_len: int = 120, device='cpu', length_norm: float = 0.7):
	model.eval()
	sos = tokenizer.sos
	eos = tokenizer.eos
	with torch.no_grad():
		# initial hypothesis list: tuples of token list and score
		hyps = [([sos], 0.0)]
		completed = []
		for step in range(max_len):
			all_candidates = []
			for seq, score in hyps:
				if seq[-1] == eos:
					completed.append((seq, score))
					continue
				#prepare input tensor
				seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
				logits = model.decode_step(seq_tensor, memory)
				#take last time step logits
				logp = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
				topk_logp, topk_ids = torch.topk(logp, beam_size)
				for k in range(topk_ids.size(0)):
					nid = int(topk_ids[k].item())
					nscore = score + float(topk_logp[k].item())
					new_seq = seq + [nid]
					all_candidates.append((new_seq, nscore))
			#keep best beam_size candidates by score
			all_candidtes.sort(key=lambda x: x[1], reverse=True)
			hyps = all_candidates[:beam_size]
			#stop early if enough completed
			if len(completed) >= beam_size:
				break
			#add remaining hyps to completed if not empty
			completed.extend(hyps)
			#apply length normalization and sort
			normalized = []
			for seq, score in completed:
				length = max(1, len(seq))
				norm_socre = score / (length ** length_norm)
				normalized.append((seq, norm_score))
			normalized.sort(key=lambda x: x[1], reverse=True)
			#decode top k sequences to token strings, drop sos and everything after eos
			results = []
			for seq, s in normalized[:beam_size]:
				#remove sos
				out_ids = seq[1:]
				#cut at first eos
				if eos in out_ids:
					out_ids = out_ids[:out_ids.index(eos)]
				tokens = tokenizer.decode(out_ids)
				results.append((' '.join(tokens), float(s)))
			return results

