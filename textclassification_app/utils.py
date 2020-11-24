import datetime


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


def print_error(msg, num_tabs=0, end="\n"):
    print(
        Specials.RED
        + "{}{} >> {}".format("\t" * num_tabs, datetime.datetime.now(), msg)
        + Specials.END,
        end=end,
    )


def print_title(msg, num_tabs=0, end="\n"):
    print(
        Specials.BOLD
        + "{}{} >> {}".format("\t" * num_tabs, datetime.datetime.now(), msg)
        + Specials.END,
        end=end,
    )


def print_message(msg, num_tabs=0, end="\n"):
    print("{}{} >> {}".format("\t" * num_tabs, datetime.datetime.now(), msg), end=end)


############################stopwords####################################
hebrew_stopwords = [
    "את",
    "זה",
    "על",
    "אבל",
    "של",
    "עם",
    "הוא",
    "אם",
    "כי",
    "יש",
    "גם",
    "או",
    "היה",
    "הזה",
    "איך",
    "שהוא",
    "שזה",
    "הייתי",
    "אחרי",
    "הם",
    "וזה",
    "עד",
    "לפני",
    "יהיה",
    "והוא",
    "וגם",
    "אחר",
    "הייתה",
    "היו",
    "ויש",
    "זו",
    "ב",
    "ה",
    "ו",
    "כ",
    "ל",
    "מ",
    "ש",
    "וה",
    "וכ",
    "ול",
    "וש",
    "וכש",
    "כש",
    "שה",
    "ולכש",
    "ולכשה",
]

