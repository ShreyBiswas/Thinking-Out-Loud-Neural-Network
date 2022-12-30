from random import randrange

from Layers import Layer, InputLayer, OutputLayer
from NodeArrays import NodeArray, InputArray, OutputArray
from ActivationFunctions import getActivationFunction
from ErrorFunctions import getErrorFunction


def mean(l: list) -> int:
    return sum(l) / len(l)


class NeuralNetwork:
    def __init__(
        self,
        inputSize: int,
        hiddenLayerSizes: list[int],
        # HYPERPARAMETERS
        outputLabels: list[str] = None,
        hiddenLayerWeights: list[list[list[float]]]  # fmt: skip
        =None,
        hiddenLayerBiases: list[list[float]]  # fmt: skip
        =None,  # includes all hidden layers + output layer as final element
        activationFunction: str = "Leaky ReLU",
        activationHyperparameters: dict = {"leakiness": 0.05},
        errorFunction: str = "MSE",
        trainingHyperparameters: dict = {
            "learningRate": 0.1,
            "learningRateDecay": 0.1,
            "epochs": 1000,
            "momentum": 0.5,
            "batchSize": 4,
        },
    ) -> None:

        self.activationFunction, self.activationDifferential = getActivationFunction(
            activationFunction, **activationHyperparameters
        )
        self.errorFunction, self.errorDifferential = getErrorFunction(errorFunction)
        self.learningRate = trainingHyperparameters.get("learningRate", 0.1)
        self.learningRateDecay = trainingHyperparameters.get("learningRateDecay", 0.1)
        self.epochs = trainingHyperparameters.get("epochs", 100)
        self.momentum = trainingHyperparameters.get("momentum", 0.5)
        self.batchSize = trainingHyperparameters.get("batchSize", 4)
        self.mean_dCost_dWeights = None
        self.mean_dCost_dBiases = None

        self.inputArray = InputArray(inputSize)
        self.inputLayer = InputLayer(
            prevNodeCount=inputSize, nextNodeCount=hiddenLayerSizes[0]
        )

        self.outputArray = OutputArray(
            size=len(outputLabels),
            index=len(hiddenLayerSizes),
            labels=outputLabels,
        )
        self.outputLayer = OutputLayer(
            indexBetween=(len(hiddenLayerSizes) - 1, len(hiddenLayerSizes)),
            prevNodeCount=hiddenLayerSizes[-1],
            nextNodeCount=len(outputLabels),
            activationFunction=self.activationFunction,
            initialWeights=hiddenLayerWeights[-1] if hiddenLayerWeights else None,
            initialBias=hiddenLayerBiases[-1] if hiddenLayerBiases else None,
        )

        self.hiddenNodeArrays = [
            NodeArray(size, index) for index, size in enumerate(hiddenLayerSizes)
        ]
        self.hiddenLayers = [
            Layer(
                indexBetween=(index, index + 1),
                prevNodeCount=hiddenLayerSizes[index],
                nextNodeCount=hiddenLayerSizes[index + 1],
                activationFunction=self.activationFunction,
                initialWeights=hiddenLayerWeights[index + 1] if hiddenLayerWeights else None  # fmt: skip
                ,
                initialBias=hiddenLayerBiases[index + 1] if hiddenLayerBiases else None,
            )
            for index in range(len(hiddenLayerSizes) - 1)
        ]

    def __str__(self) -> str:
        return "\n".join(
            str(array)
            for array in [self.inputArray, *self.hiddenNodeArrays, self.outputArray]
        )

    def __len__(self) -> int:
        return len(self.hiddenNodeArrays) + 1

    def __getitem__(self, index: int) -> NodeArray:
        # include input and output arrays
        if index == 0:
            return self.inputArray
        elif index == len(self):
            return self.outputArray
        else:
            return self.hiddenNodeArrays[index - 1]

    def __iter__(self):
        return iter([self.inputArray, *self.hiddenNodeArrays, self.outputArray])

    def loadInput(self, inputValues: list[float]):
        self.inputArray.load(inputValues)

    def loadOutput(self, actuals: list[float]):
        self.actuals = actuals

    def getOutput(self) -> list[float]:
        return self.outputArray.getValues()

    def forwardPropagate(self):
        self.inputLayer.loadInput(self.inputArray)
        self.inputLayer.forwardPropagate(self.hiddenNodeArrays[0])

        for index, layer in enumerate(self.hiddenLayers):
            # layer when index is 'i' goes between 'i-1' and 'i'
            layer.loadInput(self.hiddenNodeArrays[index])
            layer.forwardPropagate(self.hiddenNodeArrays[index + 1])

        self.outputLayer.loadInput(self.hiddenNodeArrays[-1])
        self.outputLayer.forwardPropagate(self.outputArray)

    def forwardPropagateSingle(self, inputValue: list[float], resetInput):
        self.loadInput(inputValue)
        self.forwardPropagate()
        x = self.getOutput()
        self.loadInput(resetInput)
        self.forwardPropagate()
        return x

    def getCurrentError(self) -> float:
        return self.errorFunction(self.getOutput(), self.actuals)

    def trainingStep(self, batch: tuple[list[float], list[float]]):
        # batch is a tuple of (inputValues, actuals)
        allData_dCost_dWeights = []
        allData_dCost_dBiases = []

        for inputValue, actual in batch:
            self.loadInput(inputValue)
            self.loadOutput(actual)
            self.forwardPropagate()
            (
                singleData_dCost_dWeights,
                singleData_dCost_dBiases,
            ) = self.backPropagateSingle(actual)
            allData_dCost_dWeights.append(singleData_dCost_dWeights)
            allData_dCost_dBiases.append(singleData_dCost_dBiases)

        # shape of allData_dCost_dBiases:
        # [dataIndex][layerIndex][nodeIndex]
        # shape of allData_dCost_dWeights:
        # [dataIndex][layerIndex][outputNodeIndex][inputNodeIndex]
        # remember last index is Output Layer

        # average dCost/dWeights and dCost/dBiases across all data in batch
        # first total
        mean_dCost_dWeights = allData_dCost_dWeights[0]
        mean_dCost_dBiases = allData_dCost_dBiases[0]

        for dataPiece in allData_dCost_dWeights[1:]:
            for layerIndex, layer in enumerate(dataPiece):
                for outputNodeIndex, outputNode in enumerate(layer):
                    for inputNodeIndex, inputNode in enumerate(outputNode):
                        mean_dCost_dWeights[layerIndex][outputNodeIndex][
                            inputNodeIndex
                        ] += inputNode

        for dataPiece in allData_dCost_dBiases[1:]:
            for layerIndex, layer in enumerate(dataPiece):
                for nodeIndex, node in enumerate(layer):
                    mean_dCost_dBiases[layerIndex][nodeIndex] += node

        # then divide by number of data pieces

        for layerIndex, layer in enumerate(mean_dCost_dWeights):
            for outputNodeIndex, outputNode in enumerate(layer):
                for inputNodeIndex, inputNode in enumerate(outputNode):
                    mean_dCost_dWeights[layerIndex][outputNodeIndex][
                        inputNodeIndex
                    ] /= len(batch)
        for layerIndex, layer in enumerate(mean_dCost_dBiases):
            for nodeIndex, node in enumerate(layer):
                mean_dCost_dBiases[layerIndex][nodeIndex] /= len(batch)

        if (
            self.mean_dCost_dWeights is not None
        ):  # if not first training step, use momentum
            for layerIndex, layer in enumerate(mean_dCost_dWeights):
                for outputNodeIndex, outputNode in enumerate(layer):
                    for inputNodeIndex, inputNode in enumerate(outputNode):
                        mean_dCost_dWeights[layerIndex][outputNodeIndex][
                            inputNodeIndex
                        ] += (
                            self.momentum
                            * self.mean_dCost_dWeights[layerIndex][outputNodeIndex][
                                inputNodeIndex
                            ]
                        )
            for layerIndex, layer in enumerate(mean_dCost_dBiases):
                for nodeIndex, node in enumerate(layer):
                    mean_dCost_dBiases[layerIndex][nodeIndex] += (
                        self.momentum * self.mean_dCost_dBiases[layerIndex][nodeIndex]
                    )

        # update weights and biases
        for index, layer in enumerate(self.hiddenLayers):
            layer.updateWeights(mean_dCost_dWeights[index], self.learningRate)
            layer.updateBiases(mean_dCost_dBiases[index], self.learningRate)

        self.outputLayer.updateWeights(mean_dCost_dWeights[-1], self.learningRate)
        self.outputLayer.updateBiases(mean_dCost_dBiases[-1], self.learningRate)

        # store for momentum
        self.mean_dCost_dWeights = mean_dCost_dWeights
        self.mean_dCost_dBiases = mean_dCost_dBiases

    def backPropagateSingle(self, actuals: list[float]):  # iterative solution
        # memoise dCost/dOutput
        dCost_dOutputs = [
            [None for _ in range(len(nodeArray))]
            for nodeArray in [*self.hiddenNodeArrays, self.outputArray]
        ]

        dCost_dOutputs[-1] = self.errorDifferential(self.getOutput(), actuals)

        # backPropagation from Output to final HiddenLayer
        (
            dCost_dOutputs,
            all_dCost_dWeights,
            all_dCost_dBiases,
        ) = self.backPropagateSingleFromLayer(dCost_dOutputs, hiddenLayerIndex=-1)

        allLayers_dCost_dWeights = [all_dCost_dWeights]
        allLayers_dCost_dBiases = [all_dCost_dBiases]

        # backPropagation through Hiddens
        # stops before first hidden layer, since 'prevLayer' (needed for linearOutput) is inputLayer

        for hiddenLayerIndex in range(len(self.hiddenLayers) - 1, 0, -1):
            (
                dCost_dOutputs,
                all_dCost_dWeights,
                all_dCost_dBiases,
            ) = self.backPropagateSingleFromLayer(dCost_dOutputs, hiddenLayerIndex)
            allLayers_dCost_dWeights.insert(0, all_dCost_dWeights)
            allLayers_dCost_dBiases.insert(0, all_dCost_dBiases)

        # backPropagation from first Hidden to Input
        (
            dCost_dOutputs,
            all_dCost_dWeights,
            all_dCost_dBiases,
        ) = self.backPropagateSingleFromLayer(dCost_dOutputs, hiddenLayerIndex=1)
        allLayers_dCost_dWeights.insert(0, all_dCost_dWeights)
        allLayers_dCost_dBiases.insert(0, all_dCost_dBiases)

        return allLayers_dCost_dWeights, allLayers_dCost_dBiases

    def backPropagateSingleFromLayer(self, dCost_dOutputs, hiddenLayerIndex):
        if hiddenLayerIndex == -1:
            size = len(self.outputArray)
            prevLayer = self.hiddenLayers[-1]
        elif hiddenLayerIndex == 0:
            size = len(self.hiddenNodeArrays[0])
            prevLayer = self.inputLayer
        else:
            size = len(self.hiddenNodeArrays[hiddenLayerIndex])
            prevLayer = self.hiddenLayers[hiddenLayerIndex - 1]

        all_dCost_dWeights = []
        all_dCost_dBias = []
        mean_dCost_dPrevNeurons = [[] for _ in range(size)]

        for outputNeuronIndex in range(size):
            dCost_dOutput = dCost_dOutputs[hiddenLayerIndex][
                outputNeuronIndex
            ]  # single number, how this Output Neuron affects the Cost

            dOutput_dActivation = self.activationDifferential(  # single number, how the Input Neuron's Activation affects this Output Neuron
                prevLayer.linearOutput(outputNeuronIndex)
            )

            dActivation_dWeight = (
                prevLayer.input
            )  # array of numbers, how each Weight affects this Output Neuron
            dActivation_dPrevNeuron = prevLayer.weights[
                outputNeuronIndex
            ]  # array of numbers, how each previous Neurons' value affects this Output Neuron

            dCost_dWeights = [
                dCost_dOutput * dOutput_dActivation * dAdW
                for dAdW in dActivation_dWeight
            ]
            dCost_dPrevNeurons = [
                dCost_dOutput * dOutput_dActivation * dAdPN
                for dAdPN in dActivation_dPrevNeuron
            ]

            all_dCost_dBias.append(
                dCost_dOutput * dOutput_dActivation
            )  # dActivation_dBias always equals 1

            all_dCost_dWeights.append(dCost_dWeights)
            for i in range(
                len(dCost_dPrevNeurons)
            ):  # add to corresponding input neuron tracker
                mean_dCost_dPrevNeurons[i].append(dCost_dPrevNeurons[i])

        mean_dCost_dPrevNeurons = [mean(i) for i in mean_dCost_dPrevNeurons]
        dCost_dOutputs[hiddenLayerIndex - 1] = mean_dCost_dPrevNeurons

        return dCost_dOutputs, all_dCost_dWeights, all_dCost_dBias

    def decayLearningRate(self, elapsedEpochs):
        if elapsedEpochs % 500 == 0:
            self.learningRate *= self.learningRateDecay
            self.momentum *= 1 - self.learningRateDecay


if __name__ == "__main__":
    from random import randrange, shuffle
    import random

    random.seed(1)

    hiddenLayerWeights = [
        [[randrange(-1, 1) / 10 for _ in range(2)] for _ in range(2)],
        [[randrange(-1, 1) / 10 for _ in range(3)] for _ in range(3)],
        [[randrange(-1, 1) / 10 for _ in range(3)] for _ in range(3)],
    ]
    hiddenLayerBiases = [
        [randrange(-1, 1) / 10 for _ in range(2)],
        [randrange(-1, 1) / 10 for _ in range(3)],
        [randrange(-1, 1) / 10 for _ in range(3)],
    ]
    nn = NeuralNetwork(
        inputSize=2,
        outputLabels=["a", "b", "c"],
        hiddenLayerSizes=[2, 3],
        # hiddenLayerWeights=[
        # [[1, 1], [1, 1]],
        # [[0.5, 0.5, 1], [1, 0.5, 1], [1, 0.5, 1]],
        # ],
        # hiddenLayerBiases=[[0, 0], [0, 0, 0]],
        hiddenLayerWeights=hiddenLayerWeights,
        hiddenLayerBiases=hiddenLayerBiases,
        activationFunction="Leaky ReLU",
        activationHyperparameters={"leakiness": 0.01},
        errorFunction='MSE',
        trainingHyperparameters={
            "learningRate": 0.025,
            "learningRateDecay": 0.5,
            "epochs": 2000,
            "momentum": 0.5,
            "batchSize": 4,
        },
    )
    print(nn)
    print("------")
    # nn.trainingStep([([1, 1], [2, 2, 4]), ([2, 2], [4, 4, 8]), ([3, 3], [6, 6, 12])])
    nn.loadInput([4, 4])
    # print(nn.getOutput())
    nn.loadOutput([16, 8, 8])

    def f(x):
        return [x * 4, x * 4, x * 4]

    r = 20
    Data = [([i, i], f(i)) for i in range(1, r)]
    # normalise Data
    for i in range(len(Data)):
        for j in range(len(Data[i][0])):
            Data[i][0][j] /= r
        for j in range(len(Data[i][1])):
            Data[i][1][j] /= r
    shuffle(Data)
    # find 0.2, 0.2
    for i in range(len(Data)):
        if Data[i][0] == [0.2, 0.2]:
            print(i, Data[i])
            print("----")
            break

    errors = []
    for run in range(nn.epochs):
        for i in range(nn.batchSize):
            data = [Data[i]]
            nn.trainingStep(data)
            nn.loadInput([0.2, 0.2])
            nn.loadOutput([0.8, 0.4, 0.4])

        nn.decayLearningRate(run)

        nn.loadInput([0.4, 0.4])
        nn.forwardPropagate()

        print(run, nn.getOutput(), nn.getCurrentError(), end=" ")
        if len(errors) > 10:
            if errors[-1] > errors[-2] > errors[-3]:
                print("Increases", end=" ")
        print()

        errors.append(nn.getCurrentError())

    import matplotlib.pyplot as plt

    # log scale
    plt.plot(range(nn.epochs), errors)
    plt.yscale("log")
    # show 20 ticks on the x axis
    plt.locator_params(axis="x", nbins=20)
    plt.show()
    print(nn)


# TODO: Rewrite the Neural Network to be more modular.
# TODO: Store it as a Class, with a NN.addLayer() method so each layer can be customised individually.

# TODO: Here, training isn't working. The error decreases until about 60 epochs, then increases again.
# TODO: Changing the learning rate is a temporary fix, but increasing the epochs again causes the error to increase again.
