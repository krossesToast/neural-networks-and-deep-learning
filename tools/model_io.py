import os
import sys
import numpy as np

def add_src_to_path():
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	src = os.path.join(root, "src")
	if src not in sys.path:
		sys.path.insert(0, src)
	return root

def import_network_module():
	for name in ("network", "network2", "network3"):
		try:
			return __import__(name)
		except Exception:
			pass
	raise ImportError("Could not import network module (tried network, network2, network3) from src/")

def save_model_npz(path, sizes, biases, weights):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	np.savez(
		path,
		sizes=np.array(sizes, dtype=np.int32),
		biases=np.array(biases, dtype=object),
		weights=np.array(weights, dtype=object),
	)

def load_model_npz(path):
	data = np.load(path, allow_pickle=True)
	sizes = data["sizes"].tolist()
	biases = [b for b in data["biases"]]
	weights = [w for w in data["weights"]]
	return sizes, biases, weights

def load_network_from_npz(path):
	root = add_src_to_path()
	net_mod = import_network_module()

	sizes, biases, weights = load_model_npz(path)

	net = net_mod.Network(sizes)
	net.biases = biases
	net.weights = weights
	return net