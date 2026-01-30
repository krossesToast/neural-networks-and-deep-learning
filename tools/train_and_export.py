import os
import sys
import argparse

def main():
	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	tools_dir = os.path.join(repo_root, "tools")
	if tools_dir not in sys.path:
		sys.path.insert(0, tools_dir)

	import model_io

	root = model_io.add_src_to_path()
	net_mod = model_io.import_network_module()
	mnist_loader = __import__("mnist_loader")

	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--mini_batch_size", type=int, default=10)
	parser.add_argument("--eta", type=float, default=3.0)
	args = parser.parse_args()

	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

	sizes = [784, 30, 10]
	net = net_mod.Network(sizes)
	net.SGD(training_data, args.epochs, args.mini_batch_size, args.eta, test_data=test_data)

	out_path = os.path.join(root, "models", "mnist_weights_biases.npz")
	model_io.save_model_npz(out_path, sizes, net.biases, net.weights)

	print("Saved model:", out_path)

if __name__ == "__main__":
	main()