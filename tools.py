from algebra import Matrix

def argmax(seq):
    """
    given a 1d iterable sequence, find the index of max value
    """
    return max(enumerate(seq), key=lambda val: val[1])[0]


def parse_mat(fname:str, delimiter=','):
    """
    read and parse a file to a matrix

    :return: the parsed matrix
    """
    with open(fname, 'r', encoding='utf-8') as f:
        retval = Matrix([list(map(float, line.split(delimiter))) for line in f.readlines()])
    return retval

def one_hot(inp):
    """

    :param inp:
    :return:
    """
