import sys
import json
import torch
from PIL import Image
import torchvision.transforms as T

from mnist import HandwritingCNN

MODEL_PATH = "best_model.pth"

transform = T.Compose([
	T.Resize((28, 28)),
	T.ToTensor(),
	T.Normalize((0.5,), (0.5,))
])

def predict_image(path, model):
	img = Image.open(path).convert("L")
	x = transform(img).unsqueeze(0)
	with torch.no_grad():
		out = model(x)
		return out.argmax(dim=1).item()

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
