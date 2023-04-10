# All Layer Functions
# Note: I am considering Layers to be a Mathematical Function, rather than a collection of Neurons
# This is in accordance with "Deep Learning", Goodfellow, I., et al. (2016) Deep Learning. MIT Press, Cambridge, MA. https://www.deeplearningbook.org/
# This is also in accordance with "Neural Networks and Deep Learning", Nielsen, M. (2015) Neural Networks and Deep Learning. Determination Press, Cambridge, MA. http://neuralnetworksanddeeplearning.com/

from ActivationFunctions import getActivationFunction

# Base Layer Class
class Layer:
    def __init__(
        self,
        outputSize: int,
        activationFunction=None,
        activationHyperparameters: dict = None,
        initialWeightRange: tuple[int, int] = (-5, 5),
        initialBiasRange: tuple[int, int] = (-5, 5),
    ):

        if activationFunction is None:
            self.activationFunction = lambda x: x if x > 0 else 0.1 * x  # Leaky ReLU
            self.activationDifferential = lambda x: 1 if x > 0 else 0.1
        else:
            (
                self.activationFunction,
                self.activationDifferential,
            ) = getActivationFunction(
                activationFunction, activationHyperparameters=activationHyperparameters
            )

        self.nextNodeCount = outputSize
        self.initialWeightRange = initialWeightRange
        self.initialBiasRange = initialBiasRange

    def __str__(self) -> str:
        return f"Layer {self.indexFrom} -> {self.indexTo}: [{self.prevNodeCount}] -> [{self.nextNodeCount}]"

    def __repr__(self) -> str:
        # return  all weights and biases as an equation, where input neurons are x1, x2, etc.
        string = "["
        for outputNodeIndex in range(self.nextNodeCount):
            for inputNeuronIndex in range(self.prevNodeCount):
                string += f"{self.getWeight(inputNeuronIndex, outputNodeIndex)}*x{inputNeuronIndex+1} + "
            string += f"{self.getBias(outputNodeIndex)}, "
        return f"{string[:-2]}]"

    def getArguments(self) -> tuple[int, int, int]:
        return (self.nextNodeCount, str(self.activationFunction))

    def toOutputLayer(self):
        if self is InputLayer:
            raise ValueError("Cannot convert Input Layer to Output Layer.")
        if self is OutputLayer:
            raise ValueError("Layer is already an Output Layer.")
        return OutputLayer(*self.getArguments())

    def getWeight(self, inputNodeIndex: int, outputNodeIndex: int) -> float:
        return self.weights[outputNodeIndex][inputNodeIndex]

    def getBias(self, outputNodeIndex: int) -> float:
        return self.biases[outputNodeIndex]

    def setWeights(self, newWeights: list[list[float]] = None):
        if newWeights is None:
            from random import randrange

            self.weights = [
                [randrange(-10, 10) / 100 for _ in range(self.prevNodeCount)]
                for _ in range(self.nextNodeCount)
            ]
        else:
            self.weights = newWeights

    def setBiases(self, newBias: list[float] = None):
        if newBias is None:
            from random import randrange

            self.biases = [randrange(-10, 10) / 100 for _ in range(self.nextNodeCount)]
        else:
            self.biases = newBias

    def connectBetween(self, inputFrom, outputTo):
        if self.nextNodeCount != len(outputTo):
            raise ValueError(
                f"Output size of {self.nextNodeCount} does not match next NodeArray size of {len(outputTo)}."
            )
        self.inputArray = inputFrom
        self.outputArray = outputTo
        self.prevNodeCount = len(inputFrom)
        self.nextNodeCount = len(outputTo)

        self.indexFrom = inputFrom.index
        self.indexTo = outputTo.index

    def computeOutputBetween(self, inputNodeIndex: int, outputNodeIndex: int) -> float:
        return (
            self.getWeight(inputNodeIndex, outputNodeIndex)
            * self.inputArray[inputNodeIndex]
        )

    def linearOutput(self, outputNodeIndex):
        # one run of forwardPropagate, without activation function
        return sum(
            self.computeOutputBetween(inputNeuronIndex, outputNodeIndex)
            for inputNeuronIndex in range(self.prevNodeCount)
        ) + self.getBias(outputNodeIndex)

    def forwardPropagate(self) -> list[float]:
        if self.inputArray is None:
            raise ValueError("Input Array not set.")
        if self.outputArray is None:
            raise ValueError("Output Array not set.")

        outputs = [0 for _ in range(self.nextNodeCount)]

        for outputNodeIndex in range(self.nextNodeCount):
            outputs[outputNodeIndex] = self.activationFunction(
                self.linearOutput(outputNodeIndex)
            )

        self.outputArray.load(outputs)
        return outputs

    def updateBiases(self, change: list[int], learningRate: int):
        for outputNodeIndex in range(len(change)):
            self.biases[outputNodeIndex] -= change[outputNodeIndex] * learningRate

    def updateWeights(self, change: list[list[int]], learningRate: int):
        for outputNodeIndex in range(len(change)):
            for inputNeuronIndex in range(len(change[outputNodeIndex])):
                self.weights[outputNodeIndex][inputNeuronIndex] -= (
                    change[outputNodeIndex][inputNeuronIndex] * learningRate
                )


# Input Layer Class
class InputLayer(Layer):
    def __init__(self, outputSize: int, inputSize: int):
        super().__init__(
            outputSize=outputSize,
        )
        self.inputSize = inputSize

    def __str__(self) -> str:
        return super().__str__().replace("Layer", "Input Layer")

    def connectBetween(self, inputFrom, outputTo):
        if inputFrom.type != "Input":
            raise ValueError("Input Layer must be first in network.")
        return super().connectBetween(inputFrom, outputTo)


# Output Layer Class
class OutputLayer(Layer):
    def __init__(
        self, outputSize: int, activationFunction=None, labels: list[str] = None
    ):
        super().__init__(
            outputSize=outputSize,
            activationFunction=activationFunction,
        )
        self.labels = labels

    def __str__(self) -> str:
        return super().__str__().replace("Layer", "Output Layer")

    def connectBetween(self, inputFrom, outputTo):
        if outputTo.type != "Output":
            raise ValueError("Output Layer must be last in network.")
        super().connectBetween(inputFrom, outputTo)
        self.outputArray.addLabels(self.labels)


if __name__ == "__main__":
    inputLayer = InputLayer(3, inputSize=2)
    hiddenLayer = Layer(3, "ReLU")
    outputLayer = OutputLayer(2, "ReLU")

    from NodeArrays import NodeArray, InputArray, OutputArray

    i = InputArray(2)
    h1 = NodeArray(3, 0)
    h2 = NodeArray(3, 1)
    o = OutputArray(2, 2)

    inputLayer.connectBetween(i, h1)
    hiddenLayer.connectBetween(h1, h2)
    outputLayer.connectBetween(h2, o)

    inputLayer.setWeights(
        [
            [0.15, 0.2],
            [0.25, 0.3],
            [0.35, 0.4],
        ]
    )
    inputLayer.setBiases([0.35, 0.35, 0.35])

    hiddenLayer.setWeights(
        [
            [0.4, 0.45, 0.5],
            [0.5, 0.55, 0.6],
            [0.6, 0.65, 0.7],
        ]
    )
    hiddenLayer.setBiases([0.6, 0.6, 0.6])

    outputLayer.setWeights(
        [
            [0.7, 0.75, 0.8],
            [0.8, 0.85, 0.9],
        ]
    )
    outputLayer.setBiases([0.6, 0.6])

    i.load([1, 2])
    print(repr(inputLayer))
    print(inputLayer.forwardPropagate())
    print(repr(hiddenLayer))
    print(hiddenLayer.forwardPropagate())
    print(repr(outputLayer))
    print(outputLayer.forwardPropagate())

    # try zeroing out weights
    print("---")
    i0 = InputArray(2)
    h0 = NodeArray(3, 0)
    o0 = OutputArray(3, 1)
    hL01 = Layer(3, "ReLU")
    hL02 = OutputLayer(3, "ReLU")
    hL01.connectBetween(i0, h0)
    hL02.connectBetween(h0, o0)

    hL01.setWeights([[0, 0], [0, 0], [0, 0]])
    hL01.setBiases([0, 0, 0])
    hL02.setWeights([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    hL02.setBiases([2, 1, 1])
    i0.load([1, 2])
    hL01.forwardPropagate()
    hL02.forwardPropagate()
    print(repr(hL01))
    print(repr(hL02))
    print(o0)
