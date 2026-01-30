import os
import sys
import numpy as np

def main():
	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	tools_dir = os.path.join(repo_root, "tools")
	if tools_dir not in sys.path:
		sys.path.insert(0, tools_dir)

	import model_io

	model_path = os.path.join(repo_root, "models", "mnist_weights_biases.npz")
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Missing model: {model_path}\nRun: python tools/train_and_export.py")

	net = model_io.load_network_from_npz(model_path)

	mnist_loader = __import__("mnist_loader")
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

	test_list = list(test_data)
	for i in [0, 1, 2]:
		x, y = test_list[i]
		out = net.feedforward(x)
		pred = int(np.argmax(out))
		print(f"Example {i}: true={y}, pred={pred}")

	print("Import OK ✅")

if __name__ == "__main__":
	main()