from format_data import split_data
from train_model import read_conf
import sys


def format(argc: int, argv: list) -> None:
    location = "ressources/data.csv"
    split = 0.8
    if argc >= 3:
        location = argv[2]
    if argc >= 4:
        split = float(argv[3])
    split_data(location, split)


def train(argc: int, argv: list) -> None:
    location = "configs/example.conf"
    if argc >= 3:
        location = argv[2]
    print(read_conf(location))


def error() -> int:
    print(
        "Please provide one of these modes as parameters:"
        + "'format', 'train', 'predict'\n"
        + "'format' has optional parameters 'location' and 'split'\n"
        + "'train' has optional parameter 'location'"
    )
    return 1


def main():
    argc = len(sys.argv)
    if argc < 2:
        return error()
    elif sys.argv[1] == "format":
        format(argc, sys.argv)
    elif sys.argv[1] == "train":
        train(argc, sys.argv)
    elif sys.argv[1] == "predict":
        print("coming soon")
    else:
        return error()


if __name__ == "__main__":
    main()
