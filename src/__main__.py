from format_data import split_data
from train_model import read_conf

BB = "\033[1;34m"
B = "\033[34m"
Y = "\033[33m"
R = "\033[0m"


def greet_user() -> None:
    print(f"{BB}---Welcome to Multilayer-Perceptron---")
    print(f"--------------------------------------{R}")
    print(f"{Y}Quick Guide for using the Commandline:")
    print(f"(1)   Enter your desired mode:{R} 'format', 'train' or 'predict'")
    print(f"{Y}(2.1) Additional Arguments 'format':{R} enter Path to desired data CSV")
    print(f"{Y}(2.2) Additional Arguments 'format':{R} enter desired train-test split")
    print(f"{Y}(3)   Additional Arguments 'train':{R} enter desired config file")
    print(f"{Y}(4)   Enter {R}'exit'{Y} to leave the program{R}")
    print("")


def format(argc: int, argv: list) -> None:
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
    location = "configs/example.conf"
    if argc >= 2:
        location = argv[1]
    print(read_conf(location))


def main():
    greet_user()
    while True:
        entry = input(f"{B}multilayer-perceptron:{R} ")
        args = entry.split()
        if args[0] == "format":
            format(len(args), args)
        elif args[0] == "train":
            train(len(args), args)
        elif args[0] == "predict":
            print("coming soon")
        elif args[0] == "exit":
            print("Have nice day, Bye Bye!")
            break
        else:
            print("please enter a valid mode")


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        print("Have nice day, Bye Bye!")
