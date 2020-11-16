class CrossValidation:
    def __init__(self, k_fold=5, iteration=20):
        self.k_fold = k_fold
        self.iteration = iteration

    def __str__(self):
        return str(self.k_fold) + " folds X " + str(self.iteration) + " iterations CV"
