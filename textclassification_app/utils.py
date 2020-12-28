import datetime
import logging
import sys

import os
from os.path import dirname, abspath, join

output_path = os.path.join(dirname(dirname(abspath(__file__))), "results", "output")
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

root = logging.getLogger()
stream_handler = logging.StreamHandler(stream=sys.__stdout__)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
root.addHandler(stream_handler)


file_handler = logging.FileHandler(join(output_path, "out.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
root.addHandler(file_handler)
# Create a custom logger

flask_logger = logging.getLogger("werkzeug")
file_handler = logging.FileHandler(join(output_path, "flask.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
flask_logger.addHandler(file_handler)

tf_logger = logging.getLogger("tenserflow")
file_handler = logging.FileHandler(join(output_path, "tf.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
tf_logger.addHandler(file_handler)


def print_error(msg, num_tabs=0, end="\n"):
    root.error("{} >> {}".format("\t" * num_tabs, msg))


def print_title(msg, num_tabs=0, end="\n"):
    root.debug("{} >> {}".format("\t" * num_tabs, msg))


def print_message(msg, num_tabs=0, end="\n"):
    root.info("{} >> {}".format("\t" * num_tabs, msg))

