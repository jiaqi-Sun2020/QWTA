import json
from argparse import Namespace


def args2json(args, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=4)
        file.write("\n")


def json2args(path):
    with open(path, encoding="utf-8-sig") as file:
        return Namespace(**json.load(file))
