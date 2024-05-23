def toInt(line: str):
    """return str as integer"""
    res = int(line)
    if res < 1:
        raise Exception(f"value too low: {line}")
    return res


def layer(line: str) -> list:
    """return the layers as list"""
    res = []
    args = line.split()
    for arg in args:
        res.append(toInt(arg))
    return res


def learning_rate(line: str):
    """returns learning rate"""
    res = float(line)
    if res < 0:
        raise Exception(f"value too low: {line}")
    return res


def loss(line: str):
    """returns loss function"""
    if line != "binaryCrossentropy":
        raise Exception(f"invalid loss function: {line}")
    return line


translate = {
    "layer": layer,
    "epochs": toInt,
    "loss": loss,
    "batch_size": toInt,
    "learning_rate": learning_rate,
}


res = {
    "layer": [24, 24, 24],
    "epochs": 84,
    "loss": "binaryCrossentropy",
    "batch_size": 8,
    "learning_rate": 0.0314,
}


def read_conf(location: str) -> dict:
    """opens config and parses into dict"""
    try:
        config = open(location, "r")
    except Exception as e:
        print(e)
        return res
    lines = config.readlines()
    for line in lines:
        for key in translate:
            try:
                pos = line.find(key) + len(key) + 1
                if line.find(key) != -1:
                    res[key] = translate[key](line[pos:].strip())
            except Exception as e:
                print(f"{e} chose default instead")
    return res
