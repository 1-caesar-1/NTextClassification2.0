import json
import os
from pathlib import Path

import numpy as np


def corrected_paired_students_ttest(new_score, baseline_score, k_fold, sig_level=0.05, lower_sig_level=0.01):
    """
    Calculate Corrected Paired Student's t-test according to Nadeau and Bengio's theory
    Based on the article at
    https://medium.com/analytics-vidhya/using-the-corrected-paired-students-t-test-for-comparing-the-performance-of-machine-learning-dc6529eaa97f
    :param new_score: Iterable, All results (in percentages) of the new classification that should be examined
    :param baseline_score: Iterable, All results (in percentages) of the baseline
    :param k_fold: The number of folds made in the K-fold cross-validation for the new results and for the baseline
    :param sig_level: The level of statistical significance it needs to examine
    :param lower_sig_level: A lower level of significance, will be denoted by W
    :return: "*" If the new results are significantly smaller than the baseline, "V" if they are large and nothing if
            there is no significant difference between them
    """
    # Compute the difference between the results
    diff = [y - x for y, x in zip(new_score, baseline_score)]
    # Comopute the mean of differences
    d_bar = np.mean(diff)
    # compute the variance of differences
    sigma2 = np.var(diff)
    # compute the number of data points used for training
    n1 = int(len(new_score) * ((k_fold - 1) / k_fold))
    # compute the number of data points used for testing
    n2 = int(len(new_score) / k_fold)
    # compute the total number of data points
    n = len(new_score)
    # compute the modified variance
    sigma2_mod = sigma2 * (1 / n + n2 / n1)
    # compute the t_static
    t_static = d_bar / np.sqrt(sigma2_mod)
    # compute p-value
    from scipy.stats import t
    p_value = (1 - t.cdf(np.abs(t_static), n - 1)) * 2
    # determine whether the results are significantly different, and whether they are smaller or larger
    differences_significance = ""
    if p_value <= sig_level:
        if np.mean(new_score) > np.mean(baseline_score):
            differences_significance = "V"
        else:
            differences_significance = "*"

    # later addition
    if p_value <= lower_sig_level:
        if np.mean(new_score) > np.mean(baseline_score):
            differences_significance = "W"
        else:
            differences_significance = "*"

    return differences_significance


def differences_significance(results, measure, k_folds, language, significance_level=0.05):
    # load the baseline file if exist
    dir = os.path.join(Path(__file__).parent.parent.parent, "results", "baseline", language)
    if not os.listdir(dir):
        return ""
    baseline = sorted(Path(dir).iterdir(), key=os.path.getmtime)[-1].name
    with open(os.path.join(dir, baseline), "r", encoding="utf8", errors="replace") as f:
        baseline = json.load(f)

    # find the best ML algorithm
    if measure not in baseline["results"]:
        return ""
    max = [0]
    for _, arr in baseline["results"][measure].items():
        if np.mean(arr) > np.mean(max):
            max = arr

    # convert results from fraction to percentage
    results = [x * 100 for x in results]
    baseline = [x * 100 for x in max]

    # return the corrected paired students T-test
    return corrected_paired_students_ttest(results, baseline, k_folds, significance_level)


if __name__ == "__main__":
    import numpy

    with open(r'C:\Users\natan\PycharmProjects\TextClassification2\results\json\ref results.json', 'r') as file:
        results = json.load(file)["results"]["accuracy_score"]["RandomForestClassifier"]
        print(numpy.mean(results))

    with open(r'C:\Users\natan\PycharmProjects\TextClassification2\results\baseline\hebrew\baseline.json', 'r') as file:
        baseline = json.load(file)["results"]["accuracy_score"]["RandomForestClassifier"]
        print(numpy.mean(baseline))

    # convert results from fraction to percentage
    results = [x * 100 for x in results]
    baseline = [x * 100 for x in baseline]

    # return the corrected paired students T-test
    print(corrected_paired_students_ttest(results, baseline, 5))
