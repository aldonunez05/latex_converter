import torch
import torch.onnx
import numpy as np
import sys


try:
	from train_mnist import HandwritingCNN
except ImportError:
	print("Error: couldn't import CNN")
	sys.exit(1)

def export_model():
	print('='*60)
	print("Expoting to ONNX format")
	print("="*60)

	model = HandwritingCNN()
	model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
	model.eval()
	print("model loaded successfully")
