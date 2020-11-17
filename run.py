from textclassification_app import *
from textclassification_app import main
from configuration_form import data_form


if __name__ == "__main__":
    value = input(
        "if you want to run the app enter 1 if you want to run main code only enter 2: "
    )
    if value == "1":
        data_form.run()
    else:
        main.main("configs")
