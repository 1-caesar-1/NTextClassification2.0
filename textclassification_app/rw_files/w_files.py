import json
import os
from os.path import dirname, abspath, join


def write_json_corpus(path: str, corpus: list):
    # gets a path and corpus and writes all files and jsons to path

    for post, label in corpus:
        path_txt = join(path, label["file_id"])
        path_json = path_txt.replace(".txt", ".json")
        with open(path_txt, "w", encoding="utf8", errors="replace") as txt:
            txt.write(post)
        with open(path_json, "w", encoding="utf8", errors="replace") as jsn:
            jsn.write(json.dumps(label, indent=4))


if __name__ == "__main__":
    pass
