import pickle

"Prints the pixel values of the first training image and its label"

with open("../data/mnist.pkl", "rb") as f:
    (train, val, test) = pickle.load(f, encoding="latin1")

print(len(train[0]), len(train[1]))   # number of images, number of labels
print(train[0][0])                    # first training image
print("First label:", train[1][0])    # first label