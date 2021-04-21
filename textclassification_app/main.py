import os
from multiprocessing import cpu_count
from threading import Semaphore, Thread
from typing import Callable

from alive_progress import alive_bar

from textclassification_app.classes.CrossValidation import CrossValidation
from textclassification_app.classes.Experiment import Experiment
from textclassification_app.processes.classification import classify
from textclassification_app.processes.feature_extraction_selection import extract_data
from textclassification_app.processes.normalization import normalize
from textclassification_app.processes.results_handling import (
    save_experiment_results,
    write_all_experiments,
)
from textclassification_app.processes.send_results import send_results_by_email
from textclassification_app.utils import print_title, print_message


def calc_total_iterations(experiments):
    """
    Calculation of the number of iterations that will be performed in the entire classification process
    :param experiments: List of experiments
    :return: Number of iterations (CV + Train & Test) in all experiments together
    """
    total = 0
    for experiment in experiments:
        if isinstance(experiment.classification_technique, CrossValidation):
            total += experiment.classification_technique.iteration
        else:
            total += 1
    return total


def run_experiment(experiment: Experiment, bar: Callable = None, semaphore: Semaphore = None):
    """
    The main function that receives an experiment and runs all the steps on it
    :param semaphore: The semaphore that locks the process
    :param experiment: The same experiment should be run
    :param bar: Function for updating the display. If None, the display will show nothing
    """

    if bar is None:
        def bar():
            print_message("Next iteration of " + experiment.experiment_name, num_tabs=2)
        # bar = lambda: print_message("Next iteration of " + experiment.experiment_name, num_tabs=2)

    # feature extraction & feature selection
    print_title("Extracting features")
    extract_data(experiment)

    # parameter tuning (for RF only)
    # print_title("Doing parameter tuning")
    # parameter_tuning(experiment, "GridSearchCV")

    # classification
    print_title("Classifying")
    classify(experiment, bar)
    # run_bert(experiment)
    # run_rnn(experiment)

    # write results
    print_title("Writing results")
    save_experiment_results(experiment)

    if semaphore:
        semaphore.acquire()


def main(config_path, max_threads=None):
    # create the experiments
    print_title("Creating experiments")
    experiments = [
        Experiment(config_path + "\\" + config, config.replace(".json", ""))
        for config in os.listdir(config_path)
        if config.endswith(".json") and config != "info.json"
    ]
    for experiment in experiments:
        print_message("experiment created - " + str(experiment), num_tabs=1)
    print_message("Total: " + str(len(experiments)) + " experiments", 1)

    # normalization
    print_title("Normalizing corpus")
    for experiment in experiments:
        normalize(experiment)

    # run all the experiments in different threads
    max_threads = max_threads or cpu_count()
    semaphore = Semaphore(max_threads)
    threads = []
    with alive_bar(calc_total_iterations(experiments)) as bar:
        # with threading
        for experiment in experiments:
            thread = Thread(target=run_experiment, args=(experiment, bar, semaphore))
            threads.append(thread)
            semaphore.acquire()
            thread.start()
        for thread in threads:
            thread.join()

        # without threading
        # for experiment in experiments:
        #     run_experiment(experiment, bar, semaphore)

    # write all the experiments results into Excel file
    write_all_experiments()
    send_results_by_email(["natanmanor@gmail.com", "mmgoldmeier@gmail.com"])

    print_title("Done!")


if __name__ == "__main__":
    main(r"../configs")
