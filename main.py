import os
from utils import print_title, print_message
from classes.Experiment import Experiment


def main(config_path):
    print_title("Creating experiments")
    experiments = [Experiment(config_path + '\\' + config, config.replace('.json', '')) for config in os.listdir(config_path)]
    print_message("Total: " + str(len(experiments)) + " experiments", 1)

    # normalization
    print_title("Normalizing corpus")

    # feature extraction & feature selection
    print_title("Extracting features")

    # classification
    print_title("Classifying")

    # write results
    print_title("Writing results")


if __name__ == '__main__':
    main(r'C:\Users\natan\OneDrive\שולחן העבודה\configs')