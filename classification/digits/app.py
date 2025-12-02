from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mnist import HandwritingCNN
import torch
import io
from PIL import Image
import torchvision.transforms as T

app = FastAPI()
app.add_middleware(
	CORSMiddleware,
	allow_origins["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

model = HandwritingCNN()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

transform = T.Compose([
	T.Grayscale(),
	T.Resize((28,28)),
	T.ToTensor(),
	T.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict")
async def predict(file: UploadFile):
	content = await file.read()
	img = Image.open(io.BytesIO(content))
	img = transform(img).unsqueeze(0)

	with torch.no_grad():
		logits = model(img)
		pred = logits.argmax(dim=1).item()

	return {"prediction": int(pred)}
