import json
import os
from pathlib import Path

from textclassification_app.classes.Experiment import Experiment


def save_experiment_results(experiment: Experiment):
    """
    write the result of the experiment into JSON file
    :param experiment: the experiment to save
    """
    content = {"config": experiment.config, "results": experiment.classification_results}
    parent_folder = os.path.join(Path(__file__).parent.parent.parent, "results", "json")
    path = os.path.join(parent_folder, experiment.experiment_name + " results" + ".json")
    with open(path, "w", encoding="utf8", errors="replace") as f:
        json.dump(content, f, indent=4)


def write_all_experiments():
    pass
