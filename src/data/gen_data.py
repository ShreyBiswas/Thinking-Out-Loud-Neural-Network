# no numpy
from random import randrange, shuffle


def gen_X(a, b, noise):
    return [x + randrange(-noise, noise) for x in range(a, b)]


def f(x):
    return 3 * (x**3) - 2 * (x**2) + 3 * x + 1


def gen_Y(X, noise):
    return [f(x) + randrange(-noise, noise) for x in X]


def display(X, Y):
    import matplotlib.pyplot as plt

    plt.scatter(X, Y)
    plt.show()


def save(X, Y):
    with open("src/data/sim_training_data.txt", "w") as f:
        for x, y in zip(X, Y):
            f.write(f"{x},{y}\n")


if __name__ == "__main__":
    X = gen_X(-100, 100, 10)
    shuffle(X)
    Y = gen_Y(X, 10)
    display(X, Y)
    save(X, Y)
