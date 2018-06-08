
def _parse_line_1d(line):
    values = line.split()
    label, series = values[0], values[1:]
    label = int(label)
    series = list(map(lambda x :(float(x),), series))
    return label, series


def read_input_1d(folder='input'):
    train = []
    train_filepath = '{}/{}'.format(folder, 'treino.txt')
    with open(train_filepath, 'r') as f:
        for line in f.readlines():
            entry = _parse_line_1d(line)
            train.append(entry)
    
    test = []
    test_filepath = '{}/{}'.format(folder, 'teste.txt')
    with open(test_filepath, 'r') as f:
        for line in f.readlines():
            entry = _parse_line_1d(line)
            test.append(entry)

    labels = {}
    labels_filepath = '{}/{}'.format(folder, 'rotulos.txt')
    with open(labels_filepath) as f:
        for line in f.readlines():
            label, description = line.split()
            label = int(label)
            labels[label] = description

    return train, test, labels


def _parse_line_3d(line):
    values = line.split()
    label, series = values[0], values[1:]
    label = int(label)
    series = list(map(float, series))
    series = list(zip(series[0::3], series[1::3], series[2::3]))
    return label, series


def read_input_3d(folder='input'):
    train = []
    train_filepath = '{}/{}'.format(folder, 'treino3D.txt')
    with open(train_filepath, 'r') as f:
        for line in f.readlines():
            entry = _parse_line_3d(line)
            train.append(entry)
    
    test = []
    test_filepath = '{}/{}'.format(folder, 'teste3D.txt')
    with open(test_filepath, 'r') as f:
        for line in f.readlines():
            entry = _parse_line_3d(line)
            test.append(entry)

    labels = {}
    labels_filepath = '{}/{}'.format(folder, 'rotulos3D.txt')
    with open(labels_filepath) as f:
        for line in f.readlines():
            label, description = line.split()
            label = int(label)
            labels[label] = description

    return train, test, labels