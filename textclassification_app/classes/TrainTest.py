from sklearn.model_selection import train_test_split


class TrainTest:
    def __init__(self, ratio=0.333):
        self.ratio = ratio

    def __str__(self):
        return str(round(self.ratio*100)) + "% train test split"

    def split(self, X, y):
        """
        Split arrays or matrices into random train and test subsets using sklearn train_test_split function
        """
        return train_test_split(X, y, test_size=self.ratio, random_state=42)