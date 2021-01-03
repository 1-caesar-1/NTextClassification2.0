import json
import os
from os.path import dirname, abspath


def read_json_corpus(path: str, onlyLabel: bool = False):
    # gets a path to a directory and returns a list of tuples with the post and label for each post
    docs = []
    for file in os.listdir(path):
        if file.endswith(".json") and file != "info.json":
            with open(dir + "\\" + file, "r", encoding="utf8", errors="replace") as f:
                label = {}
                if onlyLabel:
                    label = json.load(f)["classification"]
                else:
                    label = json.load(f)
            with open(
                dir + "\\" + file.replace(".json", ".txt"),
                "r",
                encoding="utf8",
                errors="replace",
            ) as f:
                data = f.read()

            docs += [(data, label)]
    return docs


def read_json_configs():
    # reads json type configs and adds the file name to the dictinory as name
    parent_dir = dirname(dirname(dirname(abspath(__file__)))) + "/configs"
    configs = []
    for config in os.listdir(parent_dir):
        if config != "info.json" and config.endswith(".json"):
            with open(
                os.path.join(parent_dir, config), "r", encoding="utf8", errors="replace"
            ) as f:
                con = json.load(f)
            con["name"] = config.replace(".json", "")
            configs.append(con)
    return configs


if __name__ == "__main__":
    read_json_configs()
