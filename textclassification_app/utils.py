import datetime
import logging
import sys

import os
from os.path import dirname, abspath, join


class Specials:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARK_CYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


# output_path = os.path.join(dirname(dirname(abspath(__file__))), "results", "output")
# formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

# root = logging.getLogger()
# stream_handler = logging.StreamHandler(stream=sys.stdout)
# stream_handler.setFormatter(formatter)
# stream_handler.setLevel(logging.DEBUG)
# root.addHandler(stream_handler)


# file_handler = logging.FileHandler(join(output_path, "out.log"))
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)
# root.addHandler(file_handler)
# Create a custom logger

# flask_logger = logging.getLogger("werkzeug")
# file_handler = logging.FileHandler(join(output_path, "flask.log"))
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)
# flask_logger.addHandler(file_handler)

# tf_logger = logging.getLogger("tenserflow")
# file_handler = logging.FileHandler(join(output_path, "tf.log"))
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)
# tf_logger.addHandler(file_handler)


def print_error(msg, num_tabs=0, end="\n"):
    logging.getLogger(__name__).error("{} >> {}".format("\t" * num_tabs, msg))
    print(
        Specials.RED
        + "{}{} >> {}".format("\t" * num_tabs, datetime.datetime.now(), msg)
        + Specials.END,
        end=end,
    )


def print_title(msg, num_tabs=0, end="\n"):
    logging.getLogger(__name__).debug("{} >> {}".format("\t" * num_tabs, msg))
    print(
        Specials.BOLD
        + "{}{} >> {}".format("\t" * num_tabs, datetime.datetime.now(), msg)
        + Specials.END,
        end=end,
    )


def print_message(msg, num_tabs=0, end="\n"):
    logging.getLogger(__name__).info("{} >> {}".format("\t" * num_tabs, msg))
    print("{}{} >> {}".format("\t" * num_tabs, datetime.datetime.now(), msg), end=end)

