import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

class HandwritingCNN(nn.Module):
	# CNN for handwriting classification
	def __init__(self):
		super(HandwritingCNN, self).__init__()

		#Convolutional layers
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

		#Pooling and dropout
		self.pool = nn.MaxPool2d(2,2)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)

		#Fully connected layers
		self.fc1 = nn.Linear(64 * 3 * 3, 128)
		self.fc2 = nn.Linear(128, 10)

		self.relu = nn.ReLU()
	
	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.pool(x)

		x = self.relu(self.conv2(x))
		x = self.pool(x)

		x = self.relu(self.conv3(x))
		x = self.pool(x)
		x = self.dropout1(x)

		# Flatten ts
		x = x.view(-1, 64 * 3 * 3)

		x = self.relu(self.fc1(x))
		x = self.dropout2(x)
		x = self.fc2(x)

		return x
	
class TrainingVisualizer:
	#To visualize training in real-time
	def __init__(self):
		self.train_losses = []
		self.val_losses = []
		self.train_accs = []
		self.val_accs = []

		plt.ion()
		self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
		self.fig.suptitle('Training Progress', fontsize=16)

	def update(self, train_loss, val_loss, train_acc, val_acc, epoch, sample_images=None, predictions=None):
		self.train_losses.append(train_loss)
		self.val_losses.append(val_loss)
		self.train_accs.append(train_acc)
		self.val_accs.append(val_acc)

		for ax in self.axes.flat:
			ax.clear()
	
		ax = self.axes[0, 0]
		ax.plot(self.train_losses, label='Train Acc', color='blue')
		ax.plot(self.val_losses, label='Val Acc', color = 'red')
		ax.set_xlabel = ('Epoch')
		ax.set_ylab = ('Loss')
		ax.set_title('Loss Over Time')
		ax.legend()
		ax.grid(True)

		ax = self.axes[0, 1]
		ax.plot(self.train_accs, label='Train Acc', color='blue')
		ax.plot(self.val_accs, label='Val Acc', color = 'red')
		ax.set_xlabel = ('Epoch')
		ax.set_ylab = ('Accuracy (%)')
		ax.set_title('Accuracy Over Time')
		ax.legend()
		ax.grid(True)

		#if sample_images is not None and predictions is not None:
			#ax = self.axes[1, 0]
			#sample_grid = sample_images[:16].cpu().numpy()
			#pred_labels = predictions[:16].cpu().numpy()

			#for i in range(min(16, len(sample_grid))):
				#plt.subplot(4, 4, i+1)
				#plt.imshow(sample_grid[i].squeeze(), cmap='gray')
				#plt.title(f'Pred: {pred_labels[i]}', fontsize=8)
				#plt.axis('off')
			#plt.sca(self.axes[1,0])
			#ax.set_title('Sample Predictions')

		ax = self.axes[1,0]
		ax.axis('off')
		metrics_text = f"""
		Epoch: {epoch + 1}

		Training:
		Loss: {train_loss:.4f}
		Accuracy: {train_acc:.2f}%

		Validation:
		Loss: {val_loss:.4f}
		Accuracy: {val_acc:.2f}%

		Best Val Acc: {max(self.val_accs):.2f}%
		"""

		ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center', family='monospace')

		ax = self.axes[1,0]
		sample_grid = sample_images[:16].cpu().numpy()
		pred_labels = predictions[:16].cpu().numpy()
		for i in range(len(sample_grid)):
			plt.imshow(sample_grid[i].squeeze(), cmap='gray')
		ax.set_title('sample prediction')

		plt.tight_layout()
		plt.pause(0.1)
	
	def save_plot(self, path):
		plt.savefig(path, dpi=150, bbox_inches='tight')
		print(f"Training ploy saved to {path}")

def train_epoch(model, train_loader, criterion, optimizer, device):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for images, labels in train_loader:
		images, labels = images.to(device), labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()

	avg_loss = running_loss / len(train_loader)
	accuracy = 100. * correct / total
	return avg_loss, accuracy
	
def validate(model, val_loader, criterion, device):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0

	sample_images = None
	predictions = None

	with torch.no_grad():
		for batch_idx, (images, labels) in enumerate(val_loader):
			images, labels = images.to(device), labels.to(device)

			outputs = model(images)
			loss = criterion(outputs, labels)

			running_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

			if batch_idx == 0:
				sample_images = images
				predictions = predicted

		avg_loss = running_loss / len(val_loader)
		accuracy = 100. * correct / total

	return avg_loss, accuracy, sample_images, predictions
	
def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"using device: {device}")

	#Hyperparams
	batch_size = 128
	num_epochs = 15
	learning_rate = 0.001

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	print("Downloading MNIST dataset...")
	train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
	val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

	print(f"Training samples: {len(train_dataset)}")
	print(f"Validation samples: {len(val_dataset)}")

	model = HandwritingCNN().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	print("\nModel Architecture:")
	print(model)
	print(f"\n Total parameters: {sum(p.numel() for p in model.parameters())}")

	visualizer = TrainingVisualizer()

	print("\nStarting training...")
	best_val_acc = 0.0

	for epoch in range(num_epochs):
		print(f"\n Epoch {epoch+1}/{num_epochs}")

		train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

		val_loss, val_acc, sample_images, predictions = validate(model, val_loader, criterion, device)

		scheduler.step()

		print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
		print(f"Val Loss: {val_loss:.4f} | Val Acc {val_acc:.2f}%")

		visualizer.update(train_loss, val_loss, train_acc, val_acc, epoch, sample_images, predictions)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), 'best_model.pth')
			print(f"New best model saved {val_acc:.2f}%")

	torch.save(model.state_dict(), 'final_model.pth')
	visualizer.save_plot('training_progress.png')

	metadata = {
		'input_shape': [1, 28, 28],
		'num_classes': 10,	
		'best_val_accuracy': best_val_acc,
		'final_val_accuracy': val_acc,
		'normalization':{
			'mean': 0.1307,
			'std': 0.3081
		}
	}

	with open('model_metadata.json', 'w') as f:
		json.dump(metadata, f, indent=2)

	print(f"\n{'='*50}")	
	print("Training complete!")
	print(f"Best validation accuracy: {best_val_acc:.2f}%")
	print(f"Final validation accuracy: {val_acc:.2f}%")
	print(f"Model saved to: best_model.pth")
	print(f"Metadata saved to: model_metadata.json")
	print(f"{'='*50}")

		
	plt.ioff()
	plt.show()

if __name__ == '__main__':
	main()

