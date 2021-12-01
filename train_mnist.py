from tools import parse_mat, batch_data
from module import Sequential, Softmax, Linear, ReLU

batch_size = 16


def get_model():
    layers = [
        Linear(128),
        ReLU(0.05),
        Linear(128),
        ReLU(0.05),
        Linear(10),
        Softmax()
    ]
    return Sequential(layers, 784, batch_size, 0.05)


def main():
    features = parse_mat('mnist/features.txt') / 127.5 - .5
    labels = parse_mat('mnist/labels.txt')
    features = batch_data(features, batch_size, True, 40)
    labels = batch_data(labels, batch_size, True, 40)
    model = get_model()
    model.fit(features, labels, 20)


if __name__ == '__main__':
    main()
