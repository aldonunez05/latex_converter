import sys
import json
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt


from mnist import HandwritingCNN

MODEL_PATH = "best_model.pth"

def preprocess(path):
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

	img = cv2.GaussianBlur(img, (5,5), 0)
	_, img = cv2.threshold(
		img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	
	kernel = np.ones((3,3), np.uint8)
	img = cv2.dilate(img, kernel, iterations=1)

	coords = cv2.findNonZero(img)
	x, y, w, h = cv2.boundingRect(coords)
	img = img[y:y+h, x:x+w]

	pad = int(max(w,h) * 0.3)
	img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

	img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
	img = img / 255.0
	img = (img-0.5) / 0.5

	return torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

transform = T.Compose([
	T.Resize((28, 28)),
	T.ToTensor(),
	T.Normalize((0.5,), (0.5,))
])

def predict_image(path, model):
	img = Image.open(path).convert("L")
	x = preprocess(path)
	
	plt.imshow(x.squeeze(), cmap="gray")
	plt.show()

	with torch.no_grad():
		logits = model(x)
		pred = logits.softmax(logits,dim=1).item()
		conf, class_idx = torch.max(pred, dim =1)
	
	return class_idx.item()

def main():
	model = HandwritingCNN()
	model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
	model.eval()

	results = {}
	for path in sys.argv[1:]:
		try:
			pred = predict_image(path, model)
			results[path] = pred
		except Exception as e:
			results[path] = str(e)
	print(json.dumps(results))

if __name__ == "__main__":
	main()
print('success')
