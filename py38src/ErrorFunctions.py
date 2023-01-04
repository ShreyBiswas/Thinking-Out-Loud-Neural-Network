def meanSquaredError(predicteds, actuals):
    return sum(
        (predicted - actual) ** 2 for predicted, actual in zip(predicteds, actuals)
    ) / len(predicteds)


def dMSE(predicteds, actuals):
    return [2 * (predicted - actual) for predicted, actual in zip(predicteds, actuals)]


def meanAbsoluteError(predicteds, actuals):
    return sum(
        abs(predicted - actual) for predicted, actual in zip(predicteds, actuals)
    ) / len(predicteds)


def dMAE(predicteds, actuals):
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
