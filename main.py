import multiprocessing
import os
import threading

from processes.classification import classify
from processes.feature_extraction_selection import extract_feature, select_features
from processes.normalization import normalize
from processes.results_handling import handle_results
from utils import print_title, print_message
from classes.Experiment import Experiment
#

def main(config_path, max_threads=None):
    # initialize the semaphore for multi-threading by the number of the cores
    if not max_threads:
        max_threads = multiprocessing.cpu_count()
    semaphore = threading.Semaphore(max_threads)

    # create the experiments
    print_title("Creating experiments")
    experiments = [Experiment(config_path + '\\' + config, config.replace('.json', '')) for config in os.listdir(config_path)]
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
        extract_feature(experiment)
        select_features(experiment)

        # classification
        print_title("Classifying")
        classify(experiment)

        # write results
        print_title("Writing results")
        handle_results(experiment)

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

    print_title("Done!")


if __name__ == '__main__':
    main(r'configs')
