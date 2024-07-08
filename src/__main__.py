from format_data import split_data
from parsing import read_conf, indexing
from train_model import Network
import matplotlib.pyplot as plt
import sys

BB = "\033[1;34m"
B = "\033[34m"
Y = "\033[33m"
R = "\033[0m"

plt.rcParams["figure.figsize"] = [20, 5]


def greet_user() -> None:
    """print tutorial for user"""
    print(f"{BB}---Welcome to Multilayer-Perceptron---")
    print(f"--------------------------------------{R}")
    print(f"{Y}Quick Guide for using the Commandline:")
    print(f"(1)   Enter your desired mode:{R} 'format', 'train' {Y}or{R} 'predict'")
    print(f"{Y}(2.1) Additional Argument 'format':{R} enter Path to desired data CSV")
    print(f"{Y}(2.2) Additional Argument 'format':{R} enter desired train-test split")
    print(f"{Y}(3)   Additional Argument 'train':{R} enter desired config file")
    print(f"{Y}(4)   Enter {R}'exit'{Y} to leave the program{R}")
    print("")


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
    print(train.shape)
    if train.shape[1] != 32 or test.shape[1] != 32:
        return print("invalid input: data must have 32 columns")
    network = Network(conf, train, test)
    network.fit()


def main():
    """main function"""
    if len(sys.argv) == 4:
        format(3, sys.argv[1:])
        train(2, sys.argv)
        return
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
            print("coming soon")
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
