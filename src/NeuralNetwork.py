from Layers import Layer, InputLayer, OutputLayer
from NodeArrays import NodeArray, InputArray, OutputArray
from ActivationFunctions import getActivationFunction
from ErrorFunctions import getErrorFunction
from utils.LayerUtils import randomiseWeights, randomiseBiases
from utils.TrainingUtils import toBatches


class NeuralNetwork:
    def __init__(
        self,
        errorFunction: str = "MSE",
        epochs: int = 100,
        batchSize: int = 10,
        learningRate: float = 0.1,
        learningRateDecay: float = 1.0,
        momentum: float = 0.0,
    ):

        self.errorFunction, self.errorDifferential = getErrorFunction(errorFunction)

        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.origLearningRate = learningRate
        self.learningRateDecay = learningRateDecay
        self.momentum = momentum
        self.origMomentum = momentum

        self.inputLayer = None
        self.outputLayer = None
        self.hiddenLayers = []

        self.compiled = False
        self.prevUpdateWeights = None
        self.prevUpdateBiases = None

    def __str__(self):
        if not self.compiled:
            return "Neural Network is not compiled yet. Please call NeuralNetwork.connectLayers() first."
        string = "\n"
        string += f"{self.inputArray}" + "\n"
        string += f"{self.inputLayer}" + "\n"
        for i in range(len(self.hiddenLayers)):
            string += f"{self.hiddenArrays[i]}" + "\n"
            string += f"{self.hiddenLayers[i]}" + "\n"
        string += f"{self.hiddenArrays[-1]}" + "\n"
        string += f"{self.outputLayer}" + "\n"
        string += f"{self.outputArray}" + "\n"
        return string

    def __repr__(self):
        if not self.compiled:
            return "Neural Network is not compiled yet. Please call NeuralNetwork.connectLayers() first."
        string = "\n"
        string += f"{self.inputArray}" + "\n"
        string += f"Input Layer -1: {repr(self.inputLayer)}" + "\n"
        for i in range(len(self.hiddenLayers)):
            string += f"{self.hiddenArrays[i]}" + "\n"
            string += f"Hidden Layer {i}: {repr(self.hiddenLayers[i])}" + "\n"
        string += f"{self.hiddenArrays[-1]}" + "\n"
        string += (
            f"Output Layer {len(self.hiddenLayers)}: {repr(self.outputLayer)}" + "\n"
        )
        string += f"{self.outputArray}" + "\n"
        return string

    def getCurrentWeights(self):
        if not self.compiled:
            raise ValueError("Neural Network is not compiled yet.")
        return [
            self.inputLayer.weights,
            *[layer.weights for layer in self.hiddenLayers],
            self.outputLayer.weights,
        ]

    def setCurrentWeights(self, weights: list[list[list[float]]]):
        if not self.compiled:
            raise ValueError("Neural Network is not compiled yet.")
        self.inputLayer.weights = weights[0]
        for i in range(len(self.hiddenLayers)):
            self.hiddenLayers[i].weights = weights[i + 1]
        self.outputLayer.weights = weights[-1]

    def addLayer(self, layer: Layer | InputLayer | OutputLayer):
        if self.inputLayer is None:
            if not isinstance(layer, InputLayer):
                raise TypeError("First layer must be an Input Layer.")
            self.inputLayer = layer
            return
        elif self.outputLayer is not None:
            raise ValueError("Output Layer already added.")
        elif isinstance(layer, OutputLayer):
            self.outputLayer = layer
            return
        else:
            self.hiddenLayers.append(layer)

    def connectLayers(
        self,
        initialWeights: list[list[list[float]]] = None,
        initialBiases: list[list[float]] = None,
    ):

        self.compiled = True

        if self.inputLayer is None:
            raise ValueError("Input Layer not added.")
        if self.outputLayer is None and len(self.hiddenLayers) == 0:
            raise ValueError("No hidden or output layers added.")

        if self.outputLayer is None:
            self.outputLayer = self.hiddenLayers.pop().toOutputLayer()

        self.inputArray = InputArray(self.inputLayer.inputSize)

        self.hiddenArrays: list[NodeArray] = [
            NodeArray(self.inputLayer.nextNodeCount, 0)
        ]

        self.hiddenArrays.extend(
            NodeArray(self.hiddenLayers[i].nextNodeCount, i + 1)
            for i in range(len(self.hiddenLayers))
        )
        self.outputArray = OutputArray(
            self.outputLayer.nextNodeCount, len(self.hiddenLayers) + 1
        )

        self.inputLayer.connectBetween(self.inputArray, self.hiddenArrays[0])
        for i in range(len(self.hiddenLayers)):
            self.hiddenLayers[i].connectBetween(
                self.hiddenArrays[i], self.hiddenArrays[i + 1]
            )
        self.outputLayer.connectBetween(self.hiddenArrays[-1], self.outputArray)

        if initialWeights is None:
            initialWeights = []
            initialWeights.append(  # add input layer weights
                randomiseWeights(
                    self.inputLayer.inputSize,
                    self.inputLayer.nextNodeCount,
                    between=self.inputLayer.initialWeightRange,
                )
            )
            initialWeights.extend(  # add hidden layer weights
                randomiseWeights(
                    self.hiddenLayers[i].prevNodeCount,
                    self.hiddenLayers[i].nextNodeCount,
                    between=self.hiddenLayers[i].initialWeightRange,
                )
                for i in range(len(self.hiddenLayers))
            )
            initialWeights.append(  # add output layer weights
                randomiseWeights(
                    self.outputLayer.prevNodeCount,
                    self.outputLayer.nextNodeCount,
                    between=self.outputLayer.initialWeightRange,
                )
            )
        if initialBiases is None:
            initialBiases = []
            initialBiases.append(
                randomiseBiases(
                    self.inputLayer.nextNodeCount,
                    between=self.inputLayer.initialBiasRange,
                )
            )
            initialBiases.extend(
                randomiseBiases(
                    self.hiddenLayers[i].nextNodeCount,
                    between=self.hiddenLayers[i].initialBiasRange,
                )
                for i in range(len(self.hiddenLayers))
            )
            initialBiases.append(
                randomiseBiases(
                    self.outputLayer.nextNodeCount,
                    between=self.outputLayer.initialBiasRange,
                )
            )

        self.setWeights(initialWeights)
        self.setBiases(initialBiases)

    def setWeights(self, weights: list[list[list[float]]]):
        # Weights Array Format:
        # [Layer][OutputNode][InputNode]

        self.inputLayer.setWeights(weights[0])
        for i in range(len(self.hiddenLayers)):
            self.hiddenLayers[i].setWeights(weights[i + 1])
        self.outputLayer.setWeights(weights[-1])

    def setBiases(self, biases: list[list[float]]):
        # Biases Array Format:
        # [Layer][OutputNode]
        self.inputLayer.setBiases(biases[0])
        for i in range(len(self.hiddenLayers)):
            self.hiddenLayers[i].setBiases(biases[i + 1])
        self.outputLayer.setBiases(biases[-1])

    def updateWeights(self, changes):
        # Changes Array Format:
        # [Layer][OutputNode][InputNode]
        self.inputLayer.updateWeights(changes[0], self.learningRate)
        for i in range(len(self.hiddenLayers)):
            self.hiddenLayers[i].updateWeights(changes[i + 1], self.learningRate)
        self.outputLayer.updateWeights(changes[-1], self.learningRate)

    def updateBiases(self, changes):
        # Changes Array Format:
        # [Layer][OutputNode]
        self.inputLayer.updateBiases(changes[0], self.learningRate)
        for i in range(len(self.hiddenLayers)):
            self.hiddenLayers[i].updateBiases(changes[i + 1], self.learningRate)
        self.outputLayer.updateBiases(changes[-1], self.learningRate)

    def forwardPropagate(self, input: list[float]) -> list[float]:
        self.inputArray.load(input)
        self.inputLayer.forwardPropagate()
        for i in range(len(self.hiddenLayers)):
            self.hiddenLayers[i].forwardPropagate()

        self.outputLayer.forwardPropagate()

        return self.outputArray.getValues()

    def predict(self, input: list[float]) -> list[float]:
        resetTo = self.inputArray.getValues()
        self.inputArray.load(input)
        prediction = self.inputLayer.forwardPropagate()

        return self.forwardPropagate(input)

    def getCurrentError(self, batch: list[tuple[list[float], list[float]]]) -> float:
        totalError = sum(
            self.errorFunction(self.predict(input), actual) for input, actual in batch
        )
        return totalError / len(batch)

    def backPropagateSingle(self, actual: list[float]) -> list[float]:

        dCost_dOutputs: list[float] = [[] for _ in range(len(self.hiddenLayers) + 2)]
        # [inputLayer, hiddenLayer1, hiddenLayer2, ..., outputLayer]

        dCost_dOutputs[-1] = self.errorDifferential(
            self.outputArray.getValues(), actual
        )  # initial dCost_dOutput from error

        (
            dCost_dOutputs,
            dCost_dWeights,
            dCost_dBiases,
        ) = self.backPropagateSingleFromLayer(actual, self.outputArray, dCost_dOutputs)

        all_dCost_dWeights: list[list[list[float]]] = [dCost_dWeights]
        all_dCost_dBiases: list[list[float]] = [dCost_dBiases]
        for array in reversed(self.hiddenArrays):
            (
                dCost_dOutputs,
                dCost_dWeights,
                dCost_dBiases,
            ) = self.backPropagateSingleFromLayer(actual, array, dCost_dOutputs)

            all_dCost_dWeights.append(dCost_dWeights)
            all_dCost_dBiases.append(dCost_dBiases)

        return (
            all_dCost_dWeights[::-1],
            all_dCost_dBiases[::-1],
        )  # reverse to get correct order from input to output

    def backPropagateSingleFromLayer(
        self,
        actual: list[float],
        outputArray: NodeArray | OutputArray,  # outputLayer is not the final layer
        # it means the layer from which this calculation of backprop is coming from
        dCost_dOutputs: list[float],
    ):

        if outputArray is self.outputArray:
            layer = self.outputLayer
        elif outputArray.index == 0:
            layer = self.inputLayer  # first hidden layer is connected to input layer
        else:
            layer = self.hiddenLayers[
                outputArray.index - 1
            ]  # each hidden layer outputs to the array with +1 index

        # positive if predicted output is too high, negative if output is too low
        dCost_dOutput = dCost_dOutputs[outputArray.index]

        dCost_dWeights = []  # format: [outputNode][inputNode]
        dCost_dBiases = []  # format: [outputNode]
        dCost_dInputs = []  # format: [outputNode][inputNode]

        for outputNode in range(len(outputArray)):
            dCost_dWeights.append([])  # element holds changes for this node
            dCost_dInputs.append([])
            dOutput_dActivation = layer.activationDifferential(
                layer.linearOutput(outputNode)
            )

            for inputNode in range(len(layer.inputArray)):
                dActivation_dWeight = layer.inputArray[inputNode]
                dActivation_dInput = layer.getWeight(inputNode, outputNode)

                dCost_dWeights[outputNode].append( # follows mathematical formula from research, dOutput and dActivation cancel
                    dCost_dOutput[outputNode]
                    * dOutput_dActivation
                    * dActivation_dWeight
                )
                dCost_dInputs[outputNode].append(
                    dCost_dOutput[outputNode] * dOutput_dActivation * dActivation_dInput
                )
            dCost_dBiases.append(  # like dCost_dWeights but with input = 1
                dCost_dOutput[outputNode] * dOutput_dActivation
            )

        # average dCost_dInputs
        dCost_dInputs = [
            (
                sum(dCost_dInputs[i][j] for i in range(len(dCost_dInputs)))
                / len(dCost_dInputs)
            )
            for j in range(len(dCost_dInputs[0]))
        ]

        dCost_dOutputs[outputArray.index - 1] = dCost_dInputs

        return dCost_dOutputs, dCost_dWeights, dCost_dBiases

    def trainingStep(self, batch: list[tuple[list[float], list[float]]]) -> None:

        # format: [layer][outputNode][inputNode]
        mean_dCost_dWeights: list[list[list[float]]] = []

        # format: [layer][outputNode]
        mean_dCost_dBiases: list[list[float]] = []

        for inputs, actuals in batch:
            self.forwardPropagate(inputs)
            dCost_dWeights, dCost_dBiases = self.backPropagateSingle(actuals)

            if not mean_dCost_dWeights:
                mean_dCost_dWeights = dCost_dWeights
                mean_dCost_dBiases = dCost_dBiases
            else:
                for i in range(len(dCost_dWeights)):
                    for j in range(len(dCost_dWeights[i])):
                        for k in range(len(dCost_dWeights[i][j])):
                            mean_dCost_dWeights[i][j][k] += dCost_dWeights[i][j][k]
                            if self.prevUpdateWeights is not None:
                                mean_dCost_dWeights[i][j][k] += (
                                    self.prevUpdateWeights[i][j][k] * self.momentum
                                )

                        # same as above but for biases
                        # can be commented out
                        # for i in range(len(dCost_dBiases)):
                        #     for j in range(len(dCost_dBiases[i])):
                        mean_dCost_dBiases[i][j] += dCost_dBiases[i][j]
                        if self.prevUpdateBiases is not None:
                            mean_dCost_dBiases[i][j] += (
                                self.prevUpdateBiases[i][j] * self.momentum
                            )

        self.prevUpdateWeights = mean_dCost_dWeights
        self.prevUpdateBiases = mean_dCost_dBiases

        self.updateWeights(mean_dCost_dWeights)
        self.updateBiases(mean_dCost_dBiases)

    def decayLearningRate(self, epochs: int):
        self.learningRate = self.origLearningRate * (
            1 / (1 + self.learningRateDecay * epochs)
        )

    def train(
        self,
        trainingData: list[tuple[list[float], list[float]]],
        testingData: list[tuple[list[float], list[float]]] = None,
    ) -> None:

        from tqdm import tqdm

        errors = []
        # testingData=  testingData[:100]
        trainingData = toBatches(trainingData, self.batchSize)

        globalMin = []
        globalMinError = float("inf")

        with tqdm(range(self.epochs)) as loop:
            for epoch in loop:
                for batch in trainingData:
                    self.trainingStep(batch)
                if testingData is not None:
                    error = self.getCurrentError(testingData)
                    errors.append(error)
                    loop.set_description(
                        f"Global Minimum Error: {globalMinError:.2f}, Error: {error:.2f}"
                    )
                self.decayLearningRate(epoch)
                if error < globalMinError:
                    globalMinError = error
                    globalMin = self.getCurrentWeights()

        self.setCurrentWeights(globalMin)
        print(f"Global Minimum Error: {globalMinError:.2f}")
        print("Using Global Minimum Weights...")

        return errors


# TODO: Keep track of the errors and the weights (needed anyways for the display) and select the minimum error weights
# TODO: Also, try messing around with the learning rate decay algorithm - maybe try a different one? Exponential?

if __name__ == "__main__":

    with open("src/data/sim_training_data.txt") as f:
        data = f.read().strip().split("\n")
    for line in range(len(data)):
        data[line] = data[line].split(",")
        data[line] = ([float(data[line][0])], [float(data[line][1])])

    # split data into training and testing
    trainingData = data[: int(len(data) * 0.8)]
    testingData = data[int(len(data) * 0.8) :]

    # split training data into batches

    nn = NeuralNetwork(
        errorFunction="MAE",
        epochs=1000,
        batchSize=10,
        learningRate=0.000075,
        learningRateDecay=0.05,
        momentum=0.05,
    )

    nn.addLayer(InputLayer(10, inputSize=1))
    nn.addLayer(Layer(20, activationFunction="Leaky ReLU"))
    nn.addLayer(Layer(10, activationFunction="Leaky ReLU"))
    nn.addLayer(Layer(1, activationFunction="Linear"))

    nn.connectLayers()

    startY = [nn.predict(sample[0]) for sample in data]

    errors = nn.train(trainingData, testingData)
    print("Training complete")
    print(f"Final error: {errors[-1]}")
    print("Plotting error graph...")

    import matplotlib.pyplot as plt

    plt.plot(errors)

    # log scale
    if errors[-1] > 1000:
        plt.yscale("log")

    # 20 ticks on x axis
    plt.xticks(range(0, len(errors), len(errors) // 20))

    plt.show()

    # generate data for graph
    X = [sample[0] for sample in data]
    trueY = [sample[1] for sample in data]
    predictedY = [nn.predict(sample[0]) for sample in data]

    # plot graph
    figure, axes = plt.subplots(1, 2)

    # pretrained
    axes[0].scatter(X, predictedY, label="Predicted", s=3)
    axes[0].scatter(X, startY, label="Start", s=3)

    # after training
    axes[1].scatter(X, trueY, label="True", s=3)
    axes[1].scatter(X, predictedY, label="Predicted", s=3)

    plt.legend()
    plt.show()