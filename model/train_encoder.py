from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

transform = transforms.Compose([
	transforms.Grayscale(),
	transforms.Resize((64, 64)),
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root="data", train=True, transform=transform, download=True)
emnist = datasets.EMNIST(root="data", split="letters", train=True, transform=transform, download=True)

from torch.utils.data import ConcatDataset
pretrain_dataset = ConcatDataset([mnist, emnist])
loader = DataLoader(pretrain_dataset, batch_size=128, shuffle=True)

encoder = CNNEncoder(out_dim=256).cuda()
clf = nn.Linear(256, 36).cuda() # 10 digits + 26 letters

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(clf.parameters()), lr=1e-3)

for epoch in range(5):
	encoder.train()
	for imgs, labels in loader:
		imgs, labels = imgs.cuda(), labels.cuda()
		with torch.no_grad():
			feats = encoder(imgs) # [B, S, D]
		feats = feats.mean(1)
		logits = clf(feats)
		loss = criterion(logits, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

torch.save(encoder.state_dict(), "encoder_pretrained.pth")
