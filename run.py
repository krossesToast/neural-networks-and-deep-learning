import src.mnist_loader as mnist_loader
import src.network as network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=3.0,
    test_data=test_data
)

mistakes = net.def_predict(test_data)
for m in mistakes[:10]:
	print(
		f"true={m['true_label']} pred={m['predicted_label']} | "
		f"out(true)={m['output_true']:.4f} out(pred)={m['output_pred']:.4f} "
		f"margin={m['margin']:.4f}"
	)

