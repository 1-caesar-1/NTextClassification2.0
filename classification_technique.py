from sklearn.model_selection import train_test_split


class TrainTest:
    def __init__(self, ratio=0.3):
        self.ratio = ratio

    def split(self, X, y):
        """
        Split arrays or matrices into random train and test subsets using sklearn train_test_split function
        """
        return train_test_split(X, y, test_size=self.ratio, random_state=42)


class CrossValidation:
    def __init__(self, k_fold=5, iteration=20):
        self.k_fold = k_fold
        self.iteration = iteration
