import os
import numpy as np
from PIL import Image

# Importiere exakt die MNIST-Pipeline
from predict_png import preprocess_mnist_style


RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_images")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "mnist_images")


def save_mnist_png(arr28, out_path):
    img = (arr28 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(out_path)


def process_all():
    for digit in range(10):
        in_dir = os.path.join(RAW_DIR, str(digit))
        out_dir = os.path.join(OUT_DIR, str(digit))
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(in_dir):
            continue

        for fname in os.listdir(in_dir):
            if not fname.lower().endswith(".png"):
                continue

            in_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname)

            x, arr_orig, arr28, stage = preprocess_mnist_style(
                in_path,
                auto_invert=True,
                show=False
            )

            save_mnist_png(arr28, out_path)
            print(f"✔ {in_path} → {out_path} [{stage}]")


if __name__ == "__main__":
    process_all()
    print("\n🎉 Alle PNGs in echtes MNIST-Format umgewandelt.")


