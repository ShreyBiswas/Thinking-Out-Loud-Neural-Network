from random import randrange


def randomiseWeights(prevNodeCount, nextNodeCount, between=(-5, 5)):
    return [
        [randrange(between[0], between[1]) for _ in range(prevNodeCount)]
        for _ in range(nextNodeCount)
    ]


def randomiseBiases(nextNodeCount, between=(-5, 5)):
    return [randrange(between[0], between[1]) for _ in range(nextNodeCount)]
