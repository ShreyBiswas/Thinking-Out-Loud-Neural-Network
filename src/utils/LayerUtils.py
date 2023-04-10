def getRandomSeed():
    x = [[0]]
    return id(x)


def randDigits(digits=1):
    root2 = 2**0.5
    num = getRandomSeed() / (1000 * root2)
    num = str(num - int(num))[2:]
    return num[:digits]


def randRange(start, stop):
    return start + (stop - start) * float(randDigits(8)) / 100000000


def randomiseWeights(
    prevNodeCount, nextNodeCount, between: tuple[float, float] = (-5, 5)
):
    return [
        [randRange(between[0], between[1]) for _ in range(prevNodeCount)]
        for _ in range(nextNodeCount)
    ]


def randomiseBiases(nextNodeCount, between: tuple[float, float] = (-5, 5)):
    return [randRange(between[0], between[1]) for _ in range(nextNodeCount)]


if __name__ == "__main__":
    print(randRange(3, 9))
    print(randomiseWeights(3, 2))
    print(randomiseBiases(2))

    print(id(0))
    print(id(0))
    print(id("asd"))
    print(id("asd"))
    print(id([[0]]))
    print(id([[0]]))
