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
		img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	)

	coords = cv2.findNonZero(img)
	x, y, w, h = cv2.boundingRect(coords)
	img = img[y:y+h, x:x+w]

	size = max(w, h)
	padded = np.zeros((size, size), dtype=np.uint8)
	padded[
		(size-h)//2:(size-h)//2+h,
		(size-w)//2:(size-w)//2+w
	] = img

	img = cv2.resize(padded, (28, 28))
	img = img / 255.0

	return torch.tensor(img).unsqueeze(0).unsqueeze(0)

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
		pred = logits.argmax(dim=1).item()
	
	return pred

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
