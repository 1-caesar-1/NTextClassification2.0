import os
import time
from datetime import datetime
from multiprocessing import cpu_count
from threading import Semaphore, Thread
from typing import Callable, Iterable

from alive_progress import alive_bar

from textclassification_app.classes.CrossValidation import CrossValidation
from textclassification_app.classes.Experiment import Experiment
from textclassification_app.classes.Watchdog import Watchdog
from textclassification_app.processes.classification import classify
from textclassification_app.processes.feature_extraction_selection import extract_data
from textclassification_app.processes.normalization import normalize
from textclassification_app.processes.parameter_tuning import parameter_tuning, ParameterTuning
from textclassification_app.processes.results_handling import (
    save_experiment_results,
    write_all_experiments,
)
from textclassification_app.processes.send_results import send_results_by_email, send_mail
from textclassification_app.utils import print_title, print_message, print_error


def calc_total_iterations(experiments):
    """
    Calculation of the number of iterations that will be performed in the entire classification process
    :param experiments: List of experiments
    :return: Number of iterations (CV + Train & Test) in all experiments together
    """
    total = 0
    for experiment in experiments:
        if isinstance(experiment.classification_technique, CrossValidation):
            total += experiment.classification_technique.iteration * len(experiment.classifiers)
        else:
            total += len(experiment.classifiers)
    return total


def watchdog_timeout_handler(to: Iterable):
    """
    A function that called when the watchdog detects an error. The function sends an email to alert about the error.
    :param to: Iterable, list of the contact to alert
    """
    print_error("The watchdog detect an error")
    subject = "An error occurred"
    body = """During the classification, an error occurred and the program hasn't continued to run for some 
    time.\nYou might want to give it a look. """
    send_mail(to, subject, body)


def handle_crush(to: Iterable, ex: Exception):
    """
    A function called when there is an exception during the program, and it sends an e-mail alert about it.
    :param to: Iterable, list of the contact to alert
    :param ex: Exception, the exception itself
    """
    subject = "An error occurred"
    body = "During the classification, an exception has been thrown. You might want to give it a " \
           "look.\n-----------------------------------------------------------\nThe exception details are as " \
           "follows:\n "
    body += str(ex)
    body += """"\n-----------------------------------------------------------"\nTotal run time: """
    body += str(datetime.now() - start)
    send_mail(to, subject, body)


def run_experiment(experiment: Experiment, bar: Callable = None, semaphore: Semaphore = None,
                   watchdog: Watchdog = None):
    """
    The main function that receives an experiment and runs all the steps on it
    :param semaphore: The semaphore that locks the process
    :param experiment: The same experiment should be run
    :param bar: Function for updating the display. If None, the display will show nothing
    :param watchdog: Optional, The watchdog that detect errors
    """

    if bar is None:
        def bar():
            print_message("Next iteration of " + experiment.experiment_name, num_tabs=2)

    # feature extraction & feature selection
    print_title("Extracting features")
    extract_data(experiment)

    # parameter tuning (for RF only)
    # print_title("Doing parameter tuning")
    # if watchdog:
    #     watchdog.stop()
    # parameter_tuning(experiment, ParameterTuning.GRID)

    # classification
    print_title("Classifying")
    classify(experiment, bar, watchdog)
    # run_rnn(experiment, bar=bar)

    # write results
    print_title("Writing results")
    save_experiment_results(experiment)

    if semaphore:
        semaphore.release()


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
    with alive_bar(calc_total_iterations(experiments)) as bar, \
            Watchdog(1800, watchdog_timeout_handler, [emails_list]) as watchdog:
        # with threading
        for experiment in experiments:
            thread = Thread(target=run_experiment, args=(experiment, bar, semaphore, watchdog))
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
    send_results_by_email(emails_list, start)

    print_title("Done!")


if __name__ == "__main__":
    emails_list = ["natanmanor@gmail.com", "mmgoldmeier@gmail.com"]
    start = datetime.now()
    try:
        main(r"../configs")
    except Exception as e:
        # in case of exception, send email with the exception details
        handle_crush(emails_list, e)
        raise e
