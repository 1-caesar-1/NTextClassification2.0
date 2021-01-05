import json
from itertools import combinations

dic = {
    "language": "Hebrew",
    "transformers": [
        "StylisticFeatures('neg1',language='Hebrew')"
    ],
    "preprocessing": [
        "lowercase"
    ],
    "features_selection": [],
    "measurements": [
        "accuracy_score"
    ],
    "classifiers": [
        "rf"
    ],
    "classification_technique": "CrossValidation()"
}

if __name__ == '__main__':
    top = 15
    size = 10
    best_families = ['fdf', 'e50th', 'huf', 'vof', 'anf', 'caf', 'mef', 'aof', 'ftf', 'acf', 'nw', 'def', 'pw', 'xte',
                     'frc', 'ref', 'vuf', 'sif', 'thf', 'lof', 'pnf', 'agf', 'te', 'slf', 'wef', 'spf', 'inf',
                     'e50tth'][:top]
    for i, comb in enumerate(combinations(best_families, size)):
        features = "StylisticFeatures(" + ",".join(["'" + str(x) + "'" for x in comb]) + ",language='Hebrew')"
        dic["transformers"] = [features]
        name = "".join([str(i)] + [str(x)[0] for x in comb]) + ".json"
        with open(name, 'w+') as f:
            json.dump(dic, f, indent=4)
        print(i+1)

    # for i in range(2, len(best_families)+1):
    #     comb = best_families[:i]
    #     features = "StylisticFeatures(" + ",".join(["'" + str(x) + "'" for x in comb]) + ",language='Hebrew')"
    #     dic["transformers"] = [features]
    #     name = "".join([str(i)] + [str(x)[0] for x in comb]) + ".json"
    #     with open(name, 'w+') as f:
    #         json.dump(dic, f, indent=4)
    #     print(i)
