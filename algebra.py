__all__ = ['e', 'RandomGenerator', 'Vector', 'Matrix', 'rand_num', 'rand_vec', 'rand_mat', 'ln']

e = 2.718281828459045


class RandomGenerator:
    """
    a class to generate random numbers withe Linear congruential generator alg
    """

    def __init__(self, a=1664525, c=1013904223, mod=2 ** 32, seed=2):
        """
        the value of a, c, mod are from book Numerical Recipes

        :param a: multiplier
        :param c: increment
        :param mod: modulus
        :param seed: seed for random generator
        """
        self.a = a
        self.c = c
        self.mod = mod
        self.seed = seed
        self.next = self.__next__

    def __iter__(self):
        return self

    def __next__(self):
        """
        get next random number

        :return: a random number ranging from 0 to 2**32
        """
        self.seed = (self.a * self.seed + self.c) % self.mod
        return self.seed


class Vector:
    """
    a class acting as a vector (treated as column vector)
    """

    def __init__(self, content: list):
        self.vec = content
        self.shape = (len(content),)
        assert all(isinstance(val, int) or isinstance(val, float) for val in content)

        self._idx = 0

    def __str__(self):
        return '[' + ' '.join(str(round(val, 8)) for val in self.vec) + ']'

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx < self.shape[0]:
            retval = self.vec[self._idx]
            self._idx += 1
            return retval
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.vec[item]

    def __setitem__(self, key, value):
        assert isinstance(value, float) or isinstance(value, int)
        self.vec[key] = value

    def clone(self):
        return Vector(self.vec[:])

    def __neg__(self):
        return Vector([-val for val in self.vec])

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        """
        element subtraction, support vector-vector, vector-number

        :return: new vector containing result
        """

        if isinstance(other, int) or isinstance(other, float):
            return Vector([other - val for val in self.vec])
        elif isinstance(other, Vector) or isinstance(other, list):
            assert len(other) == len(self.vec)
            return Vector([other[i] - self.vec[i] for i in range(len(other))])
        else:
            raise ValueError

    def __sub__(self, other):
        """
        element subtraction, support vector-vector, vector-number

        :return: new vector containing result
        """

        if isinstance(other, int) or isinstance(other, float):
            return Vector([val - other for val in self.vec])
        elif isinstance(other, Vector) or isinstance(other, list):
            assert len(other) == len(self.vec)
            return Vector([self.vec[i] - other[i] for i in range(len(other))])
        else:
            raise ValueError

    def __add__(self, other):
        """
        element addition, support vector+vector, vector+number

        :return: new vector containing result
        """

        if isinstance(other, int) or isinstance(other, float):
            return Vector([val + other for val in self.vec])
        elif isinstance(other, Vector) or isinstance(other, list):
            assert len(other) == len(self.vec)
            return Vector([self.vec[i] + other[i] for i in range(len(other))])
        else:
            raise ValueError

    def __mul__(self, other):
        """
        element multiplication, support vector*vector, vector+number

        :return: new vector containing result
        """

        if isinstance(other, int) or isinstance(other, float):
            return Vector([val * other for val in self.vec])
        elif isinstance(other, Vector) or isinstance(other, list):
            assert len(other) == len(self.vec)
            return Vector([self.vec[i] * other[i] for i in range(len(other))])
        else:
            raise ValueError

    def __matmul__(self, other):
        """
        dot product, support vector@vector

        :return: a resulting number
        """

        if isinstance(other, list) or isinstance(other, Vector):
            assert len(self.vec) == len(other)
            return sum(self.vec[i] * other[i] for i in range(len(other)))
        else:
            raise ValueError

    def __pow__(self, power):
        """
        element power, raise each element to given power

        :return: new vector containing result
        """

        assert isinstance(power, int) or isinstance(power, float)
        return Vector([val ** power for val in self.vec])

    def __rpow__(self, other):
        """
        element power, raise given number to each element of vector

        :return: new vector containing result
        """

        assert isinstance(other, int) or isinstance(other, float)
        return Vector([other ** val for val in self.vec])

    def __truediv__(self, other):
        """
        element division, support vector/vector, vector/number

        :return: new vector containing result
        """

        if isinstance(other, int) or isinstance(other, float):
            return Vector([val / other for val in self.vec])
        elif isinstance(other, Vector) and other.shape == self.shape:
            return Vector([val1 / val2 for val1, val2 in zip(self.vec, other)])


class Matrix:
    def __init__(self, content: list):
        self.mat = content
        for i in range(len(content)):
            assert isinstance(content[i], list) or isinstance(content[i], Vector)
            assert len(content[i]) == len(content[0])
            if not isinstance(content[i], Vector):
                content[i] = Vector(content[i])
        self.shape = (len(content), len(content[0]))

        self._idx = 0

    def __str__(self):
        s = ''
        for i in range(len(self.mat)):
            s += '[' if i == 0 else ' '
            s += '[' + ' '.join(str(round(val, 8)) for val in self.mat[i]) + ']'
            s += ']' if i == len(self.mat) - 1 else '\n'
        return s

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx < self.shape[0]:
            retval = self.mat[self._idx]
            self._idx += 1
            return retval
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.mat[item]

    def __setitem__(self, key, value):
        if isinstance(value, list) and len(value) == self.shape[1]:
            self.mat[key] = Vector(value)
        elif isinstance(value, Vector) and len(value) == self.shape[1]:
            self.mat[key] = value
        else:
            raise ValueError

    def clone(self):
        return Matrix([vec.clone() for vec in self.mat])

    def __neg__(self):
        return Matrix([-row for row in self.mat])

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        """
        element subtraction, support matrix-matrix, matrix-vector, matrix-number

        :return: a new matrix object containing result
        """
        if isinstance(other, int) or isinstance(other, float):
            return Matrix([other - val for val in self.mat])
        elif isinstance(other, list) or isinstance(other, Vector) or isinstance(other, Matrix):
            assert len(other) == len(self.mat)
            return Matrix([other[i] - self.mat[i] for i in range(len(self.mat))])

    def __sub__(self, other):
        """
        element subtraction, support matrix-matrix, matrix-vector, matrix-number

        :return: a new matrix object containing result
        """
        if isinstance(other, int) or isinstance(other, float):
            return Matrix([val - other for val in self.mat])
        elif isinstance(other, list) or isinstance(other, Vector) or isinstance(other, Matrix):
            assert len(other) == len(self.mat)
            return Matrix([self.mat[i] - other[i] for i in range(len(self.mat))])

    def __add__(self, other):
        """
        element addition, support matrix+matrix, matrix+vector, matrix+number

        :return: a new matrix object containing result
        """
        if isinstance(other, int) or isinstance(other, float):
            return Matrix([val + other for val in self.mat])
        elif isinstance(other, list) or isinstance(other, Vector) or isinstance(other, Matrix):
            assert len(other) == len(self.mat)
            return Matrix([self.mat[i] + other[i] for i in range(len(self.mat))])

    def __mul__(self, other):
        """
        element multiplication, support matrix*matrix, matrix*vector, matrix*number

        :return: a new matrix object containing result
        """

        if isinstance(other, int) or isinstance(other, float):
            return Matrix([val * other for val in self.mat])
        elif isinstance(other, list) or isinstance(other, Vector) or isinstance(other, Matrix):
            assert len(other) == len(self.mat)
            return Matrix([self.mat[i] * other[i] for i in range(len(self.mat))])

    def __matmul__(self, other):
        """
        matrix multiplication, only accept matrix@matrix,
        the second dimension of first matrix must equal to first dimension of second matrix

        :return: a new matrix object containing result
        """

        assert isinstance(other, Matrix) and self.shape[1] == other.shape[0]
        retval = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                s = 0
                for k in range(other.shape[0]):
                    s += self.mat[i][k] * other[k][j]
                row.append(s)
            retval.append(Vector(row))
        return Matrix(retval)

    def __pow__(self, power):
        """
        element power, each element of matrix is raised to the given power

        :return: a new matrix object containing result
        """

        return Matrix([row ** power for row in self.mat])

    def __rpow__(self, other):
        """
        element power, "other" is raised to each of the element in matrix

        :return: a new matrix object containing result
        """

        return Matrix([other ** row for row in self.mat])

    def __truediv__(self, other):
        """
        element division, each element of matrix is divided by the given value,
        support matrix/matrix, matrix/vector, matrix/number

        :return: a new matrix object containing result
        """

        if isinstance(other, Matrix) and other.shape == self.shape:
            return Matrix([row1 / row2 for row1, row2 in zip(other, self.mat)])
        elif isinstance(other, Vector) and len(other) == self.shape[0]:
            return Matrix([row / val for row, val in zip(self.mat, other)])
        else:
            return Matrix([row / other for row in self.mat])

    @property
    def T(self):
        return Matrix([list(row) for row in zip(*self.mat)])


def rand_mat(size: tuple, gen: RandomGenerator):
    """
    generate a matrix of numbers ranging from -.5 to .5

    :param size: size of returned matrix
    :param gen: random generator object
    :return: a matrix of given size
    """

    return Matrix([[rand_num(gen) for __ in range(size[1])] for _ in range(size[0])])


def rand_vec(length: int, gen: RandomGenerator):
    """
    generate a vector of numbers ranging from -.5 to .5

    :param length: length of returned vector
    :param gen: random generator object
    :return: a vector of given length
    """

    return Vector([rand_num(gen) for _ in range(length)])


def rand_num(gen: RandomGenerator):
    """
    generate a number ranging from -.5 to .5

    :param gen: random generator object
    :return: a random number
    """

    return (next(gen) / 2 ** 31 - 1.) / 2.


def ln(x):
    """
    compute the natural log of x
    """
    out_ = x - 1.
    while True:
        out = out_
        exp = e ** out
        out_ = out + 2. * (x - exp) / (x + exp)
        if abs(out - out_) <= 1e-30:
            return out_
