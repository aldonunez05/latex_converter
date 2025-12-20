import torch
import torch.nn as nn
import json
import onnx
import onnxruntime as ort

class HandwritingCNN(nn.Module):	
		def __init__(self):
				super(HandwritingCNN, self).__init__()

				self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
				self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
				self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

				self.pool = nn.MaxPool2d(2, 2)
				self.dropout1 = nn.Dropout(0.25)
				self.dropout2 = nn.Dropout(0.5)

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
				x = self.dropout(x)

				x = x.view(-1, 64 * 3 * 3)

				x = self.relu(self.fc1(x))
				x = self.dropout2(x)
				x = self.fc2(x)

				return x

def convert_to_onnx():
		device = torch.device('cpu')
		model = HandwritingCNN().to(device)

		model.load_state_dict(torch.load('best_model.pth', map_location=device))
		model.eval()

		dummy_input = torch.randn(1, 1, 28, 28, device=device)

		torch.onnx.export(
			model,
			dummy_input,
			"digital_model.onxx",
			export_params=True,
			opset_version=11,
			do_constant_folding=True,
			input_names=['input'],
			output_names=['output'],
			dynamic_axes={
				'input': {0: 'batch_size'},
				'output': {0, 'batch_size'}
			}
		)

		print("Model successfully exported to digit_model.onnx")

		onnx_model = onnx.load("digit_model.onnx")
		onnx.checker.check_model(onnx_model)
		print("ONNX model is valid")

		ort_session = ort.InferenceSession("digit_model.onnx")

		test_input = dummy_input.numpy()
		ort_inputs = {ort_session.get_inputs()[0].name: test_input}
		ort_outputs = ort_session.run(None, ort_inputs)

		with torch.no_grad():
			pytorch_output = model(dummy_input).numpy()

		print(f"PyTorch output shape: {pytorch_output.shape}")
		print(f"ONNX output shape: {ort_outputs[0].shape}")
		print(f"Max difference: {abs(pytorch_output - ort_outputs[0]).max()}")

		try:
			with open('model_metadata.json', 'r') as f:
				metadata = json.load(f)	
			metadata['onnx_file'] = 'digit_model.onnx'
			with open('model_metadata.json', 'w') as f:
				json.dump(metadata, f, indent=2)
			print("Metadata updated with ONNX file info")
		except FileNotFoundError:
			print("model_metadata.json not found, skipping metadata update")

if __name__ == '__main__':
		convert_to_onnx()

