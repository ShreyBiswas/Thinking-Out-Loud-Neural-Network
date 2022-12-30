def meanSquaredError(predicteds: list[float], actuals: list[float]):
    return sum(
        (predicted - actual) ** 2 for predicted, actual in zip(predicteds, actuals)
    ) / len(predicteds)


def dMSE(predicteds: list[float], actuals: list[float]):
    return [2 * (predicted - actual) for predicted, actual in zip(predicteds, actuals)]


def meanAbsoluteError(predicteds: list[float], actuals: list[float]):
    return sum(
        abs(predicted - actual) for predicted, actual in zip(predicteds, actuals)
    ) / len(predicteds)


def dMAE(predicteds: list[float], actuals: list[float]):
    return [
        1 if predicted > actual else -1
        for predicted, actual in zip(predicteds, actuals)
    ]


def getErrorFunction(name: str):
    errorFunctions = {
        "MSE": (meanSquaredError, dMSE),
        "MAE": (meanAbsoluteError, dMAE),
    }
    if name.upper() not in errorFunctions:
        raise ValueError(f"Error function {name} not found")
    return errorFunctions[name.upper()]
