import json

if __name__ == '__main__':
    dic = {
        "language": "Hebrew",
        "transformers": [
            "StylisticFeatures('vof','huf','aof','pnf','anf','caf','agf','frc','mef','wef','acf','fdf',language='Hebrew')"
        ],
        "preprocessing": [
            "lowercase"
        ],
        "features_selection": [],
        "measurements": [
            "accuracy_score"
        ],
        "classifiers": [
            "svc",
            "rf",
            "lr",
            "mlp",
            "mnb"
        ],
        "classification_technique": "CrossValidation()"
    }

    families = ['sep', 'pfp', 'spe', 'tpe', 'fun', 'jew', 'slg', 'war', 'trm', 'ang', 'wek', 'slp', 'sct', 'trt', 'tim',
                'xte', 'rud', 'lim', 'inf', 'sik', 'lov', 'frc', 'ref', 'acf', 'wef', 'pw', 'nw']
    for family in families:
        dic["transformers"] = [f"StylisticFeatures('{family}',language='Hebrew')"]
        with open(fr"C:\Users\natan\PycharmProjects\TextClassification2\configs\{family}.json", 'w') as file:
            json.dump(dic, file, indent=2)
