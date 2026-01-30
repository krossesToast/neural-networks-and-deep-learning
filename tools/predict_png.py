#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

def preprocess_mnist_style(png_path, invert=False, auto_invert=False, show=False):
	try:
		from PIL import Image
	except Exception as e:
		raise RuntimeError("Missing dependency Pillow. Install with: pip install pillow") from e

	if not os.path.exists(png_path):
		raise FileNotFoundError(f"PNG not found: {png_path}")

	img = Image.open(png_path).convert("L")
	arr = np.asarray(img, dtype=np.float32) / 255.0

	# Decide inversion
	mean = float(arr.mean())
	if auto_invert and mean > 0.5:
		arr = 1.0 - arr
	if invert:
		arr = 1.0 - arr

	# Threshold to find digit area (keep grayscale for later, but bbox uses mask)
	mask = arr > 0.2
	if not mask.any():
		# Nothing drawn
		x = arr.reshape(arr.shape[0] * arr.shape[1], 1)
		return x, arr, arr, "EMPTY"

	ys, xs = np.where(mask)
	y0, y1 = int(ys.min()), int(ys.max()) + 1
	x0, x1 = int(xs.min()), int(xs.max()) + 1

	crop = arr[y0:y1, x0:x1]

	# Resize cropped digit to 20x20 (MNIST-like), keep aspect ratio by padding first
	h, w = crop.shape
	side = max(h, w)
	padded = np.zeros((side, side), dtype=np.float32)
	yo = (side - h) // 2
	xo = (side - w) // 2
	padded[yo:yo+h, xo:xo+w] = crop

	# Now resize padded square to 20x20
	pil_sq = Image.fromarray((padded * 255.0).astype(np.uint8), mode="L")
	pil_20 = pil_sq.resize((20, 20))
	arr20 = np.asarray(pil_20, dtype=np.float32) / 255.0

	# Embed into 28x28 with 4px border
	arr28 = np.zeros((28, 28), dtype=np.float32)
	arr28[4:24, 4:24] = arr20

	# Center of mass shift (simple implementation without scipy)
	total = float(arr28.sum())
	if total > 0.0:
		yy, xx = np.indices(arr28.shape)
		cy = float((yy * arr28).sum() / total)
		cx = float((xx * arr28).sum() / total)
		shift_y = int(round(14 - cy))
		shift_x = int(round(14 - cx))
		arr28 = np.roll(arr28, shift_y, axis=0)
		arr28 = np.roll(arr28, shift_x, axis=1)

	x = arr28.reshape(784, 1)

	stage = f"bbox=({x0},{y0})-({x1},{y1}), mean_in={mean:.3f}"
	if show:
		try:
			import matplotlib.pyplot as plt
			plt.figure()
			plt.title("Original (normalized)")
			plt.imshow(arr, cmap="gray")
			plt.axis("off")
			plt.show()

			plt.figure()
			plt.title("Cropped digit (normalized)")
			plt.imshow(crop, cmap="gray")
			plt.axis("off")
			plt.show()

			plt.figure()
			plt.title("Final 28x28 MNIST-style")
			plt.imshow(arr28, cmap="gray")
			plt.axis("off")
			plt.show()
		except Exception:
			pass

	return x, arr, arr28, stage

def main():
	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	tools_dir = os.path.join(repo_root, "tools")
	if tools_dir not in sys.path:
		sys.path.insert(0, tools_dir)

	import model_io

	parser = argparse.ArgumentParser(description="Predict a digit from a PNG (MNIST-style preprocessing).")
	parser.add_argument("png", help="Path to PNG (any size).")
	parser.add_argument("--model", default=os.path.join(repo_root, "models", "mnist_weights_biases.npz"))
	parser.add_argument("--invert", action="store_true")
	parser.add_argument("--auto-invert", action="store_true")
	parser.add_argument("--show", action="store_true")
	args = parser.parse_args()

	if not os.path.exists(args.model):
		raise FileNotFoundError(f"Model not found: {args.model}\nRun: python tools/train_and_export.py --epochs 3")

	net = model_io.load_network_from_npz(args.model)

	x, arr_orig, arr28, stage = preprocess_mnist_style(
		args.png,
		invert=args.invert,
		auto_invert=args.auto_invert,
		show=args.show
	)

	out = np.asarray(net.feedforward(x)).reshape(-1)

	print("MODEL:", os.path.abspath(args.model))
	print("PNG:", os.path.abspath(args.png))
	print("PREPROCESS:", stage)
	print("INPUT28 stats: min=", float(arr28.min()), "max=", float(arr28.max()), "mean=", float(arr28.mean()))
	print("OUTPUT stats: min=", float(out.min()), "max=", float(out.max()), "mean=", float(out.mean()))

	top3 = np.argsort(-out)[:3]
	print("Top-3 (raw output):")
	for i in top3:
		print(f"\t{i}: {float(out[i]):.4f}")
	print("Prediction:", int(top3[0]))

if __name__ == "__main__":
	main()