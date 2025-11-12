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

# Image preprocessing
def make_transform(target_h = 128, target_w = 512):
	tf = transforms.Compose([
		transforms.Greyscale(num_output_channels=1),
		transforms.Resize((target_h, target_w)),
		transforms.ToTensor(),
		transforms.Normalize([0.5], [0.5])
	])
	return tf

# Utility to load model and tokenizer
def load_tokenizer(tok_path: str) -> Tokenizer:
	with open(tok_path, 'r', encoding='utf8') as f:
		t2i = json.load(f)
	return Tokenizer(t2i)

def load_model(checkpoint_path: str, vocab_size: int, device: str):
	model = Image2Latex(vocab_size=vocab_size, d_model=256, nhead=8, num_layers=3)
	ckpt = torch.load(checkpoint_path, map_location=device)
	if 'model_state_dict' in ckpt:
		state = ckpt['model_state_dict']
	else:
		state = ckpt
	model.load_state_dict(state)
	model.to(device)
	model.eval()
	return model

# Main predict function
def predict_image_to_latex(image_path: str, model: Image2Latex, tokenizer: Tokenizer, devide: str, beam_size: int=5):
	tf = make_transform()
	img = Image.open(image_path).convert('RGB')
	x = tf(img).unsqueeze(0).to(device)
	memory = model.encode_images(x) # [S, B, D]
	results = beam_search(model, memory, tokenizer, beam_size=beam_size, max_len=120, device=device)
	return results

# CLI Entry for now until frontend is made
def main():
	parser = argparse.ArgumentParser(description='Run inference on a single image')
	parser.add_argument('--checkpoint', required=True, help='path to model checkpoint file')
	parser.add_argument('--tokenizer', required=True, help='path to tokenizer json file mapping token to id')
	parser.add_argument('--image', required=True, help='path to input image to conver')
	parser.add_argument('--beam', type=int, default=5, help='beam size')
	args = parser.parse_args()

	#Cody ur gonna work on this more but this is the simpliest version for now
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print('Using device', device)
	
	tokenizer = load_tokenizer(args.tokenizer)
	model = load_model(args.checkpoint, vocab_size=len(tokenizer.t2i), device=device)
	results = predict_image_to_latex(args.image, model, tokenizer, device=device, beam_size=args.beam)
	print('Top predictions:')
	for i, (txt, score) in enumerate(results):
		print(f'[{i+1}] score {score:.4f} : {txt}')

if __name__ == '__main__':
	main()
