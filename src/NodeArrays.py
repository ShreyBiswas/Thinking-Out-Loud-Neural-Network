class NodeArray:
    def __init__(self, size: int, index: int) -> None:
        self.size = size
        self.values = [0 for _ in range(size)]
        self.index = index
        self.type = "Hidden"

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def load(self, values):
        if len(values) != self.size:
            raise ValueError(
                f"NodeArray {self.index} expected {self.size} values, but got {len(values)}"
            )
        self.values = list(values)

    def getValues(self):
        return self.values

    def __len__(self):
        return self.size

    def __str__(self):
        return f"NodeArray {repr(self)}"

    def __repr__(self):
        # round values to 3 decimal places
        return (
            f"{self.index}: [{', '.join([f'{round(val, 3)}' for val in self.values])}]"
        )

    def __iter__(self):
        return iter(self.values)


class InputArray(NodeArray):
    def __init__(self, size: int) -> None:
        super().__init__(size, -1)
        self.type = "Input"

    def __str__(self):
        return f"InputArray {repr(self)}"


class OutputArray(NodeArray):
    def __init__(self, size: int, index: int) -> None:
        super().__init__(size, index)
        self.type = "Output"

    def __str__(self):
        string = f"OutputArray {self.index}: | "
        return (
            string
            + " | ".join(
                [
                    f"{self.labels[i]}: {round(self.values[i], 3)}"
                    for i in range(self.size)
                ]
            )
            + " |"
        )

    def addLabels(self, labels: str = None):
        if labels is None:
            labels = [str(i) for i in range(self.size)]
        if len(labels) != self.size:
            raise ValueError(
                f"OutputArray {self.index} expected {self.size} labels, but got {len(labels)}"
            )
        self.labels = labels


if __name__ == "__main__":
    inputArray = InputArray(3)
    print(inputArray)
    inputArray.load([1, 2, 3])
    print(inputArray)
    print()

    nodeArray = NodeArray(3, 0)
    print(nodeArray)
    nodeArray.load([1, 2.1231231, 3])
    print(nodeArray)
    print(nodeArray[1])
    print()

    outputArray = OutputArray(3, 2, ["a", "b", "c"])
    print(outputArray)
    outputArray.load([1, 2.1231231, 3])
    print(outputArray)
    print()
