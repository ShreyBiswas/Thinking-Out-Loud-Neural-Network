def normalise():
    pass


def toBatches(data, batchSize):
    return [data[i : i + batchSize] for i in range(0, len(data), batchSize)]
