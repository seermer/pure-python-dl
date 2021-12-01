from algebra import Matrix


def argmax(seq):
    """
    given a 1d iterable sequence, find the index of max value
    """
    return max(enumerate(seq), key=lambda val: val[1])[0]


def parse_mat(fname: str, delimiter=','):
    """
    read and parse a file to a matrix

    :return: the parsed matrix
    """
    with open(fname, 'r', encoding='utf-8') as f:
        retval = Matrix([list(map(float, line.split(delimiter))) for line in f.readlines()])
    return retval


def batch_data(data: Matrix, batch_size: int, sample_first=False, num_batches=-1):
    """
    given a dataset, transform data into batches of data,
    by default, the given data should be in (features, sample_size)

    :param data: a dataset
    :param batch_size: currently, sample_size must be divisible by batch_size, if not, the ending data will be discarded
    :param sample_first: if true, expected shape will be (sample_size, features)
    :param num_batches: if not -1, will only return first num_batches batches
    :return: a list of batches in size (features, batch_size)
    """

    if not sample_first:
        data = data.T
    sample_size = data.shape[0]
    count = max(num_batches, sample_size//batch_size)
    # extract each sample by index to form new matrix, transpose get expected shape
    return [Matrix([data[i * batch_size + j] for j in range(batch_size)]).T for i in range(count)]
