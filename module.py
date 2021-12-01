from algebra import Matrix, Vector, RandomGenerator, rand_vec, rand_mat, ln, e

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
        exp = e ** x
        retval = []
        if self.axis == 0:
            exp = exp.T
        for item in exp:
            retval.append(item / sum(item))
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
        dw = x @ self._inp / m
        db = Vector([sum(row) / m for row in x])
        x = self.w.T @ x
        self.w = self.w - dw * self.lr
        self.b = self.b - db * self.lr
        self._inp = None
        return x


class Sequential(Modules):
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
            x = layer.forward(x)
        return x

    def backward(self, x):
        for layer in self.layers:
            x = layer.backward(x)
        return x

    # noinspection PyMethodMayBeStatic
    def loss(self, out, target):
        assert out.shape == target.shape
        return Matrix([[val1 * ln(val2) for val1, val2 in zip(row1, row2)] for row1, row2 in zip(target, out)])



    def train(self, x, target):
        out = self.forward(x, require_grad=True)
        l = self.loss(out, target)
        x = out - target
        self.backward(x)
        return l
