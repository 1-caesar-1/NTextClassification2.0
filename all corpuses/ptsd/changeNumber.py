import json
import os

pathes = [r"english\normal\HK", r"english\normal\TM", r"english\ptsd\HK", r"english\ptsd\TM", r"hebrew\normal\HK",
          r"hebrew\normal\TM", r"hebrew\ptsd\HK", r"hebrew\ptsd\TM"]

if __name__ == '__main__':
    counter = 3000
    for path in pathes:
        for file in os.listdir(path):
            if file.endswith(".json"):
                with open(os.path.join(path, file), "r", encoding="utf8", errors="replace") as jsn:
                    dict = json.load(jsn)
                dict["file_id"] = str(counter) + ".txt"
                with open(os.path.join(path, file), "w", encoding="utf8", errors="replace") as jsn:
                    json.dump(dict, jsn, indent=2)
                os.rename(os.path.join(path, file), os.path.join(path, str(counter) + ".json"))
                txt_file = file.replace(".json", ".txt")
                os.rename(os.path.join(path, txt_file), os.path.join(path, str(counter) + ".txt"))
                counter = counter + 1
