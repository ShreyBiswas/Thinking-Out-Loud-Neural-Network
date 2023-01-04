def normalise():
    pass


def toBatches(
    data: list[tuple[list[float], list[float]]], batchSize: int
) -> list[tuple[list[float], list[float]]]:
    return [data[i : i + batchSize] for i in range(0, len(data), batchSize)]
