import multiprocessing
import os
from os.path import dirname, abspath, exists
import threading
import sys
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
import tensorflow as tf
import logging


def main(config_path, max_threads=None):

    # initialize the semaphore for multi-threading by the number of the cores
    if not max_threads:
        max_threads = multiprocessing.cpu_count()
    semaphore = threading.Semaphore(max_threads)

    # create the experiments
    print_title("Creating experiments")
    experiments = [
        Experiment(config_path + "\\" + config, config.replace(".json", ""))
        for config in os.listdir(config_path)
        if config != "info.json"
    ]
    for experiment in experiments:
        print_message("experiment created - " + str(experiment), num_tabs=1)
    print_message("Total: " + str(len(experiments)) + " experiments", 1)

    # normalization
    print_title("Normalizing corpus")
    for experiment in experiments:
        normalize(experiment)

    def run_experiment(experiment: Experiment):
        # feature extraction & feature selection
        print_title("Extracting features")
        extract_data(experiment)

        # classification
        print_title("Classifying")
        classify(experiment)

        # write results
        print_title("Writing results")
        save_experiment_results(experiment)

        # update the semaphore
        semaphore.release()

    # run all the experiments in different threads
    threads = []
    for experiment in experiments:
        thread = threading.Thread(target=run_experiment, args=(experiment,))
        threads.append(thread)
        semaphore.acquire()  # start the thread only if the semaphore is available
        thread.start()

    # wait for all threads
    for thread in threads:
        thread.join()

    # write all the experiments results into Excel file
    write_all_experiments()
    send_results_by_email(["natanmanor@gmail.com", "mmgoldmeier@gmail.com"])

    print_title("Done!")


if __name__ == "__main__":
    main(r"../configs")
