import json

if __name__ == '__main__':
    dic = {"language": "Hebrew",
           "transformers": [
               "TfidfVectorizer(max_features=500,analyzer='word',lowercase=False,ngram_range=(1,1),use_idf=True,min_df=2)"
           ],
           "preprocessing": [],
           "features_selection": [],
           "measurements": [
               "accuracy_score"
           ],
           "classifiers": [
               "mlp",
               "svc",
               "lr",
               "rf",
               "mnb"
           ],
           "classification_technique": "CrossValidation()"
           }

    for i in range(500, 6000, 500):
        trans = f"TfidfVectorizer(max_features={i},analyzer='word',lowercase=False,ngram_range=(1,1),use_idf=True,min_df=2)"
        dic["transformers"] = [trans]
        with open(r"C:\Users\nmanor\PycharmProjects\TextClassification2.0\configs\\" + str(i) + ".json", "w") as file:
            json.dump(dic, file, indent=4)
