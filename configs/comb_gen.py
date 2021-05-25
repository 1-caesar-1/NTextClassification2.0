import itertools
import json
import uuid

if __name__ == '__main__':
    dic = {
        "language": "English",
        "transformers": [],
        "preprocessing": ["lowercase"],
        "features_selection": [],
        "measurements": ["accuracy_score"],
        "classifiers": ["rf"],
        "classification_technique": "CrossValidation()"
    }

    families = {'e50te', 'thf', 'caf', 'inf', 'vof', 'aof', 'acf', 'anf', 'nw', 'agf', 'lof', 'frc', 'spe', 'pnf',
                'sxf', 'slf', 'skf', 'pw'}
    max_size = 18
    min_size = 18

    total = 0
    for i, size in enumerate(range(min_size, max_size + 1)):
        for comb in itertools.combinations(families, size):
            total += 1
            dic["transformers"] = [
                "StylisticFeatures(" + ",".join("'" + family + "'" for family in comb) + ",language='English')"]
            file = open(r"../configs/" + str(size) + str(uuid.uuid4()) + ".json", "w")
            json.dump(dic, file, indent=4)
            file.close()

    print("Total =", total)
