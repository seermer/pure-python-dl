from algebra import Matrix, Vector, RandomGenerator, rand_vec, rand_mat, ln, e
from tools import argmax

__all__ = ['Modules', 'ReLU', 'Softmax', 'Linear', 'Sequential']


class Modules:
    def build(self, in_units: int, batch_size: int, lr: float, gen: RandomGenerator):
        raise NotImplementedError

    def forward(self, x, require_grad=False):  # x is output from previous layer, (inp_len, batch_size)
        raise NotImplementedError

    def backward(self, x):  # x is derivative propagated from next layer
        raise NotImplementedError


class ReLU(Modules):
    def __init__(self, alpha=0.):
        self.alpha = alpha
        self._inp = None

    def build(self, in_units: int, batch_size: int, lr: float, gen: RandomGenerator):
        pass

    def forward(self, x, require_grad=False):
        if require_grad:
            self._inp = x.clone()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i][j] = max(x[i][j], self.alpha * x[i][j])
        return x

    def backward(self, x):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i][j] = x[i][j] if self._inp[i][j] > 0 else x[i][j] * self.alpha
        return x


class Softmax(Modules):
    def __init__(self, axis=0):  # axis=0 is per column, 1 is per row
        self.axis = axis

    def build(self, in_units: int, batch_size: int, lr: float, gen: RandomGenerator):
        pass

    def forward(self, x, require_grad=False):
        m = max(max(row) for row in x)
        exp = e ** (x - m)  # prevent overflow by subtracting max
        if self.axis == 0:
            exp = exp.T
        retval = [item / sum(item) for item in exp]
        return Matrix(retval) if self.axis == 1 else Matrix(retval).T

    def backward(self, x):  # instead of chain rule, compute with analytic derivative combined with loss
        return x


class Linear(Modules):
    def __init__(self, units):
        self.units = units
        self.w = None
        self.b = None
        self.lr = None
        self._inp = None

    def build(self, in_units: int, batch_size: int, lr: float, gen: RandomGenerator):
        self.lr = lr
        self.w = rand_mat((self.units, in_units), gen)
        self.b = rand_vec(self.units, gen)

    def forward(self, x, require_grad=False):
        if require_grad:
            self._inp = x.clone()
        return self.w @ x + self.b

    def backward(self, x):
        m = self._inp.shape[1]  # batch size
        dw = x @ self._inp.T / m
        db = Vector([sum(row) / m for row in x])
        x = self.w.T @ x
        self.w = self.w - dw * self.lr
        self.b = self.b - db * self.lr
        self._inp = None
        return x


class Sequential(Modules):
    """
    a sequential model
    """

    def __init__(self, layers: list, in_units: int, batch_size: int, lr: float, gen=None):
        self.layers = layers
        self.units = None
        self.build(in_units, batch_size, lr, gen if gen is not None else RandomGenerator())

    def build(self, in_units: int, batch_size: int, lr: float, gen: RandomGenerator):
        for layer in self.layers:
            layer.build(in_units, batch_size, lr, gen)
            in_units = layer.units if hasattr(layer, 'units') else in_units
        self.units = in_units

    def forward(self, x, require_grad=False):
        for layer in self.layers:
            x = layer.forward(x, require_grad)
            print('forward', layer)
        return x

    def backward(self, x):
        for layer in self.layers:
            x = layer.backward(x)
            print('backward', layer)
        return x

    # noinspection PyMethodMayBeStatic
    def _loss(self, out, target):
        assert out.shape == target.shape
        return sum(sum(val1 * ln(val2) for val1, val2 in zip(row1, row2)) for row1, row2 in zip(target, out))

    # noinspection PyMethodMayBeStatic
    def _count_correct(self, out, target):
        """
        count the number of correct predictions

        :return: correct number of predictions
        """

        correct = 0
        for out_val, target_val in zip(zip(*out), zip(*target)):
            if argmax(out_val) == argmax(target_val):
                correct += 1
        return correct

    def train(self, x, target):
        """
        train on a single batch of data, expect shape (features, batch_size)

        :returns: total loss, correct predictions
        """

        out = self.forward(x, require_grad=True)
        l = self._loss(out, target)
        correct = self._count_correct(out, target)
        x = out - target
        self.backward(x)
        return l, correct

    def fit(self, x, target, epochs):
        """
        train on a dataset (multiple batches)

        note that for both x and target, each sample is expected on a column (which means shape[1] == batch_size)

        :param x: a list or iterable of multiple batches, each batch expected as (features, batch_size)
        :param target: a list or iterable of targets(labels) corresponding to each batch of x, expect one-hot labels
        :param epochs: number of epochs to train on given dataset
        """

        assert len(x) == len(target)

        print('training starts....')
        for i in range(1, epochs + 1):
            curr_l, curr_c, curr_sample = 0, 0, 0
            print('epoch', i, end=': ')
            for batch_x, batch_y in zip(x, target):
                assert batch_x.shape[1] == batch_y.shape[1]
                l, c = self.train(batch_x, batch_y)
                curr_l += l
                curr_c += c
                curr_sample += batch_y.shape[1]
                print('acc', curr_c / curr_sample, '| loss', curr_l / curr_sample)
