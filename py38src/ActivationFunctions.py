class Linear:
    def __init__(self, **kwargs):
        self.Linear = self.__call__
        self.differential = lambda x: 1

    def __call__(self, x):
        return x

    def __str__(self):
        return "Linear"

    def __repr__(self):
        return "Linear()"


class ReLU:
    def __init__(self, **kwargs):
        self.ReLU = self.__call__
        self.differential = lambda x: 1 if x >= 0 else 0

    def __call__(self, x):
        return max(0, x)

    def __str__(self):
        return "ReLU"

    def __repr__(self):
        return "ReLU()"


class LeakyReLU:
    def __init__(self, **kwargs):
        # kwargs is {leakiness: float}
        self.leakiness = kwargs.get("leakiness", 0.1)
        self.LeakyReLU = self.__call__
        self.differential = lambda x: 1 if x >= 0 else self.leakiness

    def __call__(self, x):
        return max(self.leakiness * x, x)

    def __str__(self):
        return "Leaky ReLU"

    def __repr__(self):
        return "LeakyReLU()"


class Sigmoid:
    def __init__(self, **kwargs):
        from math import exp

        self.exp = exp
        self.Sigmoid = self.__call__
        self.differential = lambda x: self(x) * (1 - self(x))

    def __call__(self, x):
        try:
            return 1 / (1 + self.exp(-x))
        except OverflowError:  # x is too large
            return 1 if x > 0 else 0

    def __str__(self):
        return "Sigmoid"

    def __repr__(self):
        return "Sigmoid()"


class Tanh:
    def __init__(self, **kwargs):
        from math import exp

        self.exp = exp
        self.Tanh = self.__call__
        self.differential = lambda x: 1 - self(x) ** 2

    def __call__(self, x):
        return (self.exp(x) - self.exp(-x)) / (self.exp(x) + self.exp(-x))

    def __str__(self):
        return "Tanh"

    def __repr__(self):
        return "Tanh()"


def getActivationFunction(name: str, **kwargs):
    activationFunctions = {
        "LINEAR": Linear,
        "RELU": ReLU,
        "LEAKY RELU": LeakyReLU,
        "SIGMOID": Sigmoid,
        "TANH": Tanh,
    }
    if name.upper() not in activationFunctions:
        raise ValueError(f"Activation function {name} not found.")

    func = activationFunctions[name.upper()](**kwargs)
    return (func, func.differential)


if __name__ == "__main__":

    relu, drelu = getActivationFunction("ReLU")
    leaky_relu, dleaky_relu = getActivationFunction("Leaky ReLU", leakiness=0.01)
    print(relu(1))
    print(leaky_relu(1))
    print(drelu(1))
    print(dleaky_relu(1))

    print(relu(-1))
    print(leaky_relu(-1))
    print(drelu(-1))
    print(dleaky_relu(-1))
