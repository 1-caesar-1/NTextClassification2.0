from textclassification_app import *
from textclassification_app import main
from configuration_form import data_form
import logging
import sys
import os
from os.path import dirname, abspath, join


def set_log():
    output_path = os.path.join(dirname(abspath(__file__)), "results", "output")
    con_formatter = logging.Formatter("%(asctime)s %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    )
    logging.getLogger().handlers = []
    # main logger
    # stream handler to print all logs to regular stdout
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(con_formatter)
    stream_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(stream_handler)

    file_handler = logging.FileHandler(join(output_path, "out.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    # Create a custom logger
    # flask logger
    logging.getLogger("werkzeug")
    file_handler = logging.FileHandler(join(output_path, "flask.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logging.getLogger("werkzeug").addHandler(file_handler)

    # tenserflow logger
    file_handler = logging.FileHandler(join(output_path, "tf.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logging.getLogger("tenserflow").addHandler(file_handler)


if __name__ == "__main__":
    set_log()
    value = input(
        "if you want to run the app enter 1 if you want to run main code only enter 2: "
    )
    if value == "1":
        data_form.run()
    else:
        main.main("configs")
