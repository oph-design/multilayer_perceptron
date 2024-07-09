import os
import numpy as np
import matplotlib.pyplot as plt
from format_data import split_data
from parsing import read_conf, read_data, indexing
from network import train_factory, predict_factory
from network import Network


BB = "\033[1;34m"
B = "\033[34m"
Y = "\033[33m"
R = "\033[0m"

plt.rcParams["figure.figsize"] = [20, 5]
plt.tight_layout()


def greet_user() -> None:
    """print tutorial for user"""
    print(f"{BB}---Welcome to Multilayer-Perceptron---")
    print(f"--------------------------------------{R}")
    print(f"{Y}Quick Guide for using the Commandline:")
    print(f"(1)   Enter a command:{R} 'format' 'predict' 'train' 'show' 'config'")
    print(f"{Y}(2.1) Additional Argument 'format':{R} Path to desired Data CSV")
    print(f"{Y}(2.2) Additional Argument 'format':{R} desired train-test split")
    print(f"{Y}(3.1) Additional Argument 'predict':{R} desired model file")
    print(f"{Y}(3.2) Additional Argument 'predict':{R} Path to desired Data CSV")
    print(f"{Y}(4)   Additional Argument 'train':{R} desired config file")
    print(f"{Y}(5)   Enter {R}'exit'{Y} to leave the program{R}")
    print("")


def show() -> None:
    files = [f for f in os.listdir("results/models")]
    files = [os.path.splitext(f)[0] for f in files]
    print("available models:")
    for file in files:
        print(file)


def config() -> None:
    config = {}
    name = input("name: ")
    config["layer"] = input("layer: ")
    config["epochs"] = input("epochs: ")
    config["loss"] = input("loss: ")
    config["batch_size"] = input("batch_size: ")
    config["learning_rate"] = input("learning_rate: ")
    with open("configs/" + name + ".conf", "w") as file:
        for key, value in config.items():
            file.write(f"{key}: {value}\n")


def format(argc: int, argv: list) -> None:
    """uses the format program and adjusts the parameters"""
    location = "ressources/data.csv"
    split = 0.8
    if argc >= 2:
        location = argv[1]
    if argc >= 3:
        try:
            split = float(argv[2])
        except Exception as e:
            print(e)
    split_data(location, split)


def train(argc: int, argv: list) -> None:
    """executes training algorithm"""
    location = "configs/example.conf"
    if argc >= 2:
        location = argv[1]
    conf = read_conf(location)
    train = indexing("datasets/data_train.csv", conf["batch_size"])
    test = indexing("datasets/data_test.csv", conf["batch_size"])
    if train.shape[1] != 32 or test.shape[1] != 32:
        return print("invalid input: data must have 32 columns")
    dim = np.insert([30, 2], 1, conf["layer"])
    layers = train_factory(dim, conf["batch_size"])
    network = Network(train, test, layers, conf)
    network.fit()
    network.save_to_file()


def predict(argc: int, argv: list) -> None:
    data_name = "datasets/data_test.csv"
    if argc < 2 or not os.path.exists("results/models/" + argv[1] + ".npz"):
        return print("please provide an existing model as argument")
    if argc > 2:
        data_name = argv[2]
    data = read_data(data_name)
    model = dict(np.load("results/models/" + argv[1] + ".npz"))
    if data.shape[1] != 31:
        return print("invalid input: data must have 32 columns")
    layers = predict_factory(model, data.shape[0])
    network = Network(data, data, layers, None)
    network.evalulate_model(argv[1])


def main():
    """main function"""
    greet_user()
    while True:
        entry = input(f"{B}multilayer-perceptron:{R} ")
        args = entry.split()
        if len(args) == 0:
            continue
        elif args[0] == "format":
            format(len(args), args)
        elif args[0] == "train":
            train(len(args), args)
        elif args[0] == "predict":
            predict(len(args), args)
        elif args[0] == "show":
            show()
        elif args[0] == "config":
            config()
        elif args[0] == "exit":
            print("Have a nice day, Bye Bye!")
            break
        else:
            print("please enter a valid mode")


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        print("Have a nice day, Bye Bye!")
