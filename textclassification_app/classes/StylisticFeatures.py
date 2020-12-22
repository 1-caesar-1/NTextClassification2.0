import re

from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from textclassification_app.classes.stopwords_and_lists import negative_list_hebrew, time_expressions_hebrew, \
    positive_list_hebrew, time_expressions_english, first_person_expressions_hebrew, first_person_expressions_english, \
    second_person_expressions_hebrew, second_person_expressions_english, third_person_expressions_hebrew, \
    third_person_expressions_english, anorexia_family, anorexia_family_en, food_family, food_family_en, fat_family, \
    fat_family_en, ana_family, ana_family_en, hunger_family, hunger_family_en, me_family, me_family_en, vomiting_family, \
    vomiting_family_en, pain_family, pain_family_en, anger_family, anger_family_en, sleep_family, sleep_family_en, \
    sport_family, sport_family_en, thinness_family_en, thinness_family, calories_family_en, calories_family, \
    vulgarity_family, vulgarity_family_en, decreasing_family, decreasing_family_en, increasing_family, \
    increasing_family_en, sickness_family, sickness_family_en, love_family, love_family_en, \
    extended_time_expressions_hebrew, extended_time_expressions_english, noun_family, sex_family_en, cursing_family_en, \
    alcohol_family_en, smoke_family_en, export_50_terms_he, export_50_terms_en, export_50_terms_trans_he, \
    export_50_terms_trans_en, increasing_expressions_hebrew, increasing_expressions_english, \
    decreasing_expressions_hebrew, decreasing_expressions_english, doubt_expressions_hebrew, doubt_expressions_english, \
    emotion_expressions_hebrew, emotion_expressions_english, inclusion_expressions_hebrew, \
    inclusion_expressions_english, power1_expressions_hebrew, power1_expressions_english, power2_expressions_hebrew, \
    power2_expressions_english, power3_expressions_hebrew, power3_expressions_english, powerm1_expressions_hebrew, \
    powerm1_expressions_english, powerm2_expressions_hebrew, powerm2_expressions_english, powerm3_expressions_hebrew, \
    powerm3_expressions_english, powerm4_expressions_hebrew, powerm4_expressions_english


def StylisticFeatures(*names: str, language: str):
    """
    This function returns FeatureUnion for all the stylistic features it gets as parameters or a single features if
    it gets a single property.
    The function knows how to deal with 4 different types of properties:

    1. Words list (e.g. "inf")
        In this case the function will put into the FeatureUnion two transformers, one of which is a regular
        TfidfVectorizer and one of them is the ratio of the words in the list to all the other words in the document.
        Note that the second transformer will not be added in any situation.
    2. Set of features (e.g. "wef")
        In this case the function will run recursively for each feature from the set, eventually connecting the results
        of the sub-class to the final FeatureUnion.
        This option is intended for defined sets of features that are typically called as a group, for example "wef".
    3. Special transformer (e.g. "rfc")
        Sometimes it is necessary to collect several features with similar operation (like counting common characters).
        This option will add a special transformer to the FeatureUnion by calling the function that will create the
        desired transformer. This transformer will be activated by the final FeatureUnion.
    4. Specific functions (e.g. "cc")
        Sometimes it is necessary to calculate a single feature (for example, the average number of words in a post).
        In this case, you will take an existing function that knows how to receive the corpus and its language, and
        wrap it in a special transformer that knows how to activate this function while activating the transform
        function of that transformer.

    Note that in each of the above cases all that has to be done is to pass to function StylisticFeatures the names of
    the features.
    The full list of features is in the dictionary in the initialize_features_dict function.
    :param names: list of names of the features
    :param language: the language of the repository on which the transformer will run ("hebrew" or "english")
    :return: FeatureUnion of the wanted features
    """
    language = language.lower()
    stylistic_features_dict = initialize_features_dict(language)
    vectorizers = []

    for name in names:
        name = name.lower()
        # option 1:
        # if the feature is list, add TfidfVectorizer with known vocabulary
        if isinstance(stylistic_features_dict[name], list):
            lst = set(stylistic_features_dict[name])
            vectorizers += [(name + '1', TfidfVectorizer(vocabulary=lst))]
            # if needed, add the percentage of the list of words from all the other words
            if name != "e50th" and name != "e50te" and name != "e50tth" and name != "e50tte":
                vectorizers += [
                    (name + '2', StylisticFeaturesTransformer(stylistic_features_dict[name], name, language))]

        # option 2:
        # if the current feature is set of feature (e.g. "acf") add the transformers recursively
        elif isinstance(stylistic_features_dict[name], set):
            for group_feature in stylistic_features_dict[name]:
                vectorizers += [(group_feature, StylisticFeatures(group_feature, language=language))]

        # option 3:
        # if the current feature is frc, add the InitTransformer of FRC
        elif name == "frc":
            vectorizers += [(name, stylistic_features_dict[name]())]

        # else, add the StylisticFeaturesTransformer of the feature
        else:
            vectorizers += [(name, StylisticFeaturesTransformer(stylistic_features_dict[name], name, language))]

    # return FeatureUnion off all the stylistic features or the stylistic features itself if there is only one
    return FeatureUnion(vectorizers, n_jobs=-1) if len(vectorizers) > 1 else vectorizers[0][1]


def initialize_features_dict(language):
    stylistic_features_dict = {
        "cc": chars_count,
        "wc": words_count,
        "sc": sentence_count,
        "emc": exclamation_mark_count,
        "qsmc": question_mark_count,
        "scc": special_characters_count,
        "qtmc": quotation_mark_count,
        "alw": average_letters_word,
        "als": average_letters_sentence,
        "aws": average_words_sentence,
        "awl": average_word_length,
        "ie": increasing_expressions,
        "dex": decreasing_expressions,
        "nw": negative_list_hebrew if language == "hebrew" else negative_words,
        "pw": positive_list_hebrew if language == "hebrew" else positive_words,
        "te": time_expressions_hebrew if language == "hebrew" else time_expressions_english,
        "de": doubt_expressions,
        "ee": emotion_expressions,
        "fpe": first_person_expressions_hebrew if language == "hebrew" else first_person_expressions_english,
        "spe": second_person_expressions_hebrew if language == "hebrew" else second_person_expressions_english,
        "tpe": third_person_expressions_hebrew if language == "hebrew" else third_person_expressions_english,
        "ine": inclusion_expressions,
        "p1": power1,
        "p2": power2,
        "p3": power3,
        "pm1": power_minus1,
        "pm2": power_minus2,
        "pm3": power_minus3,
        "pm4": power_minus4,
        "ap": all_powers,
        "frc": known_repeated_chars,
        "rc": repeated_chars,
        "dw": doubled_words,
        "tw": tripled_words,
        "dh": doubled_hyphen,
        "dx": doubled_exclamation,
        "tx": tripled_exclamation,
        "ww": words_wealth,
        "owc": once_words,
        "twc": twice_words,
        "ttc": three_times_words,
        "aof": anorexia_family if language == "hebrew" else anorexia_family_en,
        "fdf": food_family if language == "hebrew" else food_family_en,
        "ftf": fat_family if language == "hebrew" else fat_family_en,
        "anf": ana_family if language == "hebrew" else ana_family_en,
        "huf": hunger_family if language == "hebrew" else hunger_family_en,
        "mef": me_family if language == "hebrew" else me_family_en,
        "vof": vomiting_family if language == "hebrew" else vomiting_family_en,
        "pnf": pain_family if language == "hebrew" else pain_family_en,
        "agf": anger_family if language == "hebrew" else anger_family_en,
        "slf": sleep_family if language == "hebrew" else sleep_family_en,
        "spf": sport_family if language == "hebrew" else sport_family_en,
        "thf": thinness_family if language == "hebrew" else thinness_family_en,
        "caf": calories_family if language == "hebrew" else calories_family_en,
        "vuf": vulgarity_family if language == "hebrew" else vulgarity_family_en,
        "def": decreasing_family if language == "hebrew" else decreasing_family_en,
        "inf": increasing_family if language == "hebrew" else increasing_family_en,
        "sif": sickness_family if language == "hebrew" else sickness_family_en,
        "lof": love_family if language == "hebrew" else love_family_en,
        "xte": extended_time_expressions_hebrew if language == "hebrew" else extended_time_expressions_english,
        "nof": noun_family,
        "sxf": sex_family_en,
        "cuf": cursing_family_en,
        "alf": alcohol_family_en,
        "skf": smoke_family_en,
        "e50th": export_50_terms_he,
        "e50te": export_50_terms_en,
        "e50tth": export_50_terms_trans_he,
        "e50tte": export_50_terms_trans_en,
        "acf": {"wc", "cc", "sc", "alw", "als", "aws", "awl"},
        "ref": {"dw", "tw", "dh", "dx", "tx"},
        "wef": {"ww", "owc", "twc", "ttc"}
    }
    return stylistic_features_dict


class StylisticFeaturesTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, featurizers, feature, language):
        self.language = language
        self.featurizers = featurizers
        self.feature_name = feature

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        return self

    def transform(self, X):
        """Given a list of original data, return a list of feature vectors."""
        if isinstance(self.featurizers, list):
            return csr_matrix(general_list(X, self.featurizers))

        _X = self.featurizers(X, self.language)
        return csr_matrix(_X)

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        if isinstance(self.featurizers, list):
            return [self.feature_name]
        return self.featurizers("get feature names", self.language)


def prevalence_rate(str, lst, length_relation=False):
    """
    :param str: the text that needs to be analyzed
    :param lst: list the words that need to check the prevalence of inside the text
    :param length_relation: do attach importance to the length of each words in the list
    :return: the percentage of repetition of the number of words in the list of all words in the text
    """
    orginal_str = str
    num = 0
    for word in sorted(set(lst), key=len, reverse=True):
        if length_relation:
            length = len(word.split(" "))
        else:
            length = 1
        num += str.lower().count(word) * length
        str = str.replace(word, "")
    return [num / len(re.findall(r"\b\w[\w-]*\b", orginal_str.lower()))]


# -----------------------------------------------------------------------------------------
# Quantitative features
# (Normalized in words and characters)


def chars_count(data, language):
    """
    1
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: list the number of characters in each post
    """
    if data == "get feature names":
        return ["chars count"]

    return [[len(post)] for post in data]


def words_count(data, language):
    """
    2
    :param data: the corpus
    :param language: the language of the data (ignored)
    :return: list the number of words in each post
    """
    if data == "get feature names":
        return ["words count"]

    return [[len(re.findall(r"\b\w[\w-]*\b", post.lower()))] for post in data]


def sentence_count(data, language):
    """
    3
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: list the estimated number of sentences in each post
    """
    if data == "get feature names":
        return ["sentence count"]

    for post in data:
        post.replace("...", ".").replace("..", ".")
        post.replace("!!!", "!").replace("!!", "!")
        post.replace("???", "?").replace("??", "?")
    return [[len(re.split(r"[.!?]+", post))] for post in data]


def exclamation_mark_count(data, language):
    """
    4
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: list of number of repetitions of ! normalized by the number of characters in each post
    """
    if data == "get feature names":
        return ["exclamation mark count"]

    return [[post.count("!") / len(post)] for post in data]


def question_mark_count(data, language):
    """
    5
    :param data: the corpus
    :param language: the language of the data (ignored)
    :return: list of number of repetitions of ? normalized by the number of characters in each post
    """
    if data == "get feature names":
        return ["question mark count"]

    return [[post.count("?") / len(post)] for post in data]


def special_characters_count(data, language):
    """
    6
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: list of number of repetitions of special characters normalized by the number of characters in each post
    """
    if data == "get feature names":
        return ["special characters count"]

    result = [0] * len(data)
    for char in ["@", "#", "$", "&", "*", "%", "^"]:
        for i in range(len(data)):
            result[i] += data[i].count(char)
    return [[result[i] / len(data[i])] for i in range(len(data))]


def quotation_mark_count(data, language):
    """
    7
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: list of number of repetitions of " or ' normalized by the number of characters in each post
    """
    if data == "get feature names":
        return ["quotation mark count"]

    result = [0] * len(data)
    for char in ['"', "'"]:
        for i in range(len(data)):
            result[i] += data[i].count(char)
    return [[result[i] / len(data[i])] for i in range(len(data))]


# -----------------------------------------------------------------------------------------
# Averages features


def average_letters_word(data, language):
    """
    8
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: list of the average length of a words per post
    """

    def average_per_post(post):
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        num = 0
        for word in post:
            num += len(word)
        return num / len(post)

    if data == "get feature names":
        return ["average letters word"]

    return [[average_per_post(post)] for post in data]


def average_letters_sentence(data, language):
    """
    9
    :param data: the corpus
    :param language: the language of the data (ignored)
    :return: list the estimated average of the length of each sentence (no spaces)
    """

    def average_per_post(post):
        post = re.split(r"[.!?]+", post.replace(" ", ""))
        num = 0
        for sentence in post:
            num += len(sentence)
        return num / len(post)

    if data == "get feature names":
        return ["average letters sentence"]

    return [[average_per_post(post)] for post in data]


def average_words_sentence(data, language):
    """
    10
    :param data: the corpus
    :param language: the language of the data (ignored)
    :return: list the estimated average of the num of words in each sentence
    """

    def average_per_post(post):
        post = re.split(r"[.!?]+", post)
        num = 0
        for sentence in post:
            num += len(re.findall(r"\b\w[\w-]*\b", sentence.lower()))
        return num / len(post)

    if data == "get feature names":
        return ["average words sentence"]

    return [[average_per_post(post)] for post in data]


def average_word_length(data, language):
    """
    :param data: the corpus
    :param language: the language of the data (ignored)
    :return: list the average words length in each post
    """
    if data == "get feature names":
        return ["average word length"]

    new_list = []
    for post in data:
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        sum = 0
        for word in post:
            sum += len(word)
        new_list += [[sum / len(post)]]
    return new_list


# -----------------------------------------------------------------------------------------
# Reduction and increase features
# (Normalized in the number of words)


def increasing_expressions(data, language):
    """
    11 - 12 - 13
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of increasing words out of the total words in each post
    """

    lst = {"hebrew": increasing_expressions_hebrew, "english": increasing_expressions_english}[language]

    if data == "get feature names":
        return ["increasing expressions"]

    return [prevalence_rate(post, lst, True) for post in data]


def decreasing_expressions(data, language):
    """
    14 - 15 - 16
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of decreasing words out of the total words in each post
    """

    lst = {
        "hebrew": decreasing_expressions_hebrew,
        "english": decreasing_expressions_english,
    }[language]

    if data == "get feature names":
        return ["decreasing expressions"]

    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Positive and negative features
# (Normalized in the number of words)


def negative_words(data, language):
    """
    17 - 18 - 19
    Determine the language of the text and enable the appropriate function
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of positive words out of the total words in each post
    """

    # English version: negative words
    def english_negative_words(data):
        def negative_count(post):
            sid = SentimentIntensityAnalyzer()
            neg_word_num = 0
            post = re.findall(r"\b\w[\w-]*\b", post.lower())
            for word in post:
                if (sid.polarity_scores(word)["compound"]) <= -0.5:
                    neg_word_num += 1
            return [neg_word_num / len(post)]

        return [negative_count(post) for post in data]

    # Hebrew version: negative words
    def hebrew_negative_words(data):
        return [prevalence_rate(post, negative_list_hebrew) for post in data]

    if data == "get feature names":
        return ["negative words"]

    if language == "hebrew":
        return hebrew_negative_words(data)
    # nltk.download('vader_lexicon')
    return english_negative_words(data)


def positive_words(data, language):
    """
    20 - 21 - 22
    Determine the language of the text and activate the appropriate function
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of positive words out of the total words in each post
    """

    # English version: positive words
    def english_positive_words(data):
        def positive_count(post):
            sid = SentimentIntensityAnalyzer()
            pos_word_num = 0
            post = re.findall(r"\b\w[\w-]*\b", post.lower())
            for word in post:
                if (sid.polarity_scores(word)["compound"]) >= 0.5:
                    pos_word_num += 1
            return [pos_word_num / len(post)]

        return [positive_count(post) for post in data]

    # Hebrew version: positive words
    def hebrew_positive_words(data):
        return [prevalence_rate(post, positive_list_hebrew) for post in data]

    if data == "get feature names":
        return ["positive words"]

    if language == "hebrew":
        return hebrew_positive_words(data)
    # nltk.download('vader_lexicon')
    return english_positive_words(data)


# -----------------------------------------------------------------------------------------
# Time features
# (Normalized in the number of words)
def time_expressions(data, language):
    """
    23 - 24 - 25
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of time words out of the total words in each post
    """
    lst = {"hebrew": time_expressions_hebrew, "english": time_expressions_english}[language]

    if data == "get feature names":
        return ["time expressions"]

    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Features of disapproval and doubt
# (Normalized in the number of words)
def doubt_expressions(data, language):
    """
    26 - 27 - 28
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of doubt words out of the total words in each post
    """
    lst = {"hebrew": doubt_expressions_hebrew, "english": doubt_expressions_english}[language]

    if data == "get feature names":
        return ["doubt expressions"]

    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Features of emotions
# (Normalized in the number of words)
def emotion_expressions(data, language):
    """
    29
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of emotion terms out of the total words in each post
    """
    lst = {"hebrew": emotion_expressions_hebrew, "english": emotion_expressions_english}[language]

    if data == "get feature names":
        return ["emotion expressions"]

    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Features of persons
# (Normalized in the number of words)


def first_person_expressions(data, language):
    """
    30
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of first person terms out of the total words in each post
    """
    lst = {"hebrew": first_person_expressions_hebrew, "english": first_person_expressions_english}[language]

    if data == "get feature names":
        return ["first person expressions"]

    return [prevalence_rate(post, lst, False) for post in data]


def second_person_expressions(data, language):
    """
    31
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of second person terms out of the total words in each post
    """
    lst = {"hebrew": second_person_expressions_hebrew, "english": second_person_expressions_english}[language]

    if data == "get feature names":
        return ["second person expressions"]

    return [prevalence_rate(post, lst, False) for post in data]


def third_person_expressions(data, language):
    """
    32
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of third person terms out of the total words in each post
    """
    lst = {"hebrew": third_person_expressions_hebrew, "english": third_person_expressions_english}[language]

    if data == "get feature names":
        return ["third person expressions"]

    return [prevalence_rate(post, lst, False) for post in data]


# -----------------------------------------------------------------------------------------
# Features of inclusion
# (Normalized in the number of words)


def inclusion_expressions(data, language):
    """
    33
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of inclusion terms out of the total words in each post
    """
    lst = {"hebrew": inclusion_expressions_hebrew, "english": inclusion_expressions_english}[language]

    if data == "get feature names":
        return ["inclusion expressions"]

    return [prevalence_rate(post, lst, False) for post in data]


# -----------------------------------------------------------------------------------------
# Features of powers
# (Normalized in the number of words)


def power1(data, language):
    """
    34
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of terms from power 1 out of the total words in each post
    """
    lst = {"hebrew": power1_expressions_hebrew, "english": power1_expressions_english}[language]

    if data == "get feature names":
        return ["power 1"]

    return [prevalence_rate(post, lst, True) for post in data]


def power2(data, language):
    """
    35
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of terms from power 2 out of the total words in each post
    """
    lst = {"hebrew": power2_expressions_hebrew, "english": power2_expressions_english}[language]

    if data == "get feature names":
        return ["power 2"]

    return [prevalence_rate(post, lst, True) for post in data]


def power3(data, language):
    """
    36
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of terms from power 3 out of the total words in each post
    """
    lst = {"hebrew": power3_expressions_hebrew, "english": power3_expressions_english}[language]

    if data == "get feature names":
        return ["power 3"]

    return [prevalence_rate(post, lst, True) for post in data]


def power_minus1(data, language):
    """
    37
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of terms from power -1 out of the total words in each post
    """
    lst = {"hebrew": powerm1_expressions_hebrew, "english": powerm1_expressions_english}[language]

    if data == "get feature names":
        return ["power -1"]

    return [prevalence_rate(post, lst, True) for post in data]


def power_minus2(data, language):
    """
    38
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of terms from power -2 out of the total words in each post
    """
    lst = {"hebrew": powerm2_expressions_hebrew, "english": powerm2_expressions_english}[language]

    if data == "get feature names":
        return ["power -2"]

    return [prevalence_rate(post, lst, True) for post in data]


def power_minus3(data, language):
    """
    39
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of terms from power -3 out of the total words in each post
    """
    lst = {"hebrew": powerm3_expressions_hebrew, "english": powerm3_expressions_english}[language]

    if data == "get feature names":
        return ["power -3"]

    return [prevalence_rate(post, lst, True) for post in data]


def power_minus4(data, language):
    """
    40
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of terms from power -4 out of the total words in each post
    """
    lst = {"hebrew": powerm4_expressions_hebrew, "english": powerm4_expressions_english}[language]

    if data == "get feature names":
        return ["power -4"]

    return [prevalence_rate(post, lst, True) for post in data]


def all_powers(data, language):
    """
    41
    :param language: the language of the data
    :param data: the corpus
    :return: list the percentage of terms from all powers out of the total words in each post
    """
    lst = []
    if language == "hebrew":
        lst += (power1_expressions_hebrew + power2_expressions_hebrew + power3_expressions_hebrew)
        lst += (
                    powerm1_expressions_hebrew + powerm2_expressions_hebrew + powerm3_expressions_hebrew + powerm4_expressions_hebrew)
    else:
        lst += (
                power1_expressions_english
                + power2_expressions_english
                + power3_expressions_english
        )
        lst += (
                powerm1_expressions_english
                + powerm2_expressions_english
                + powerm3_expressions_english
                + powerm4_expressions_english
        )

    if data == "get feature names":
        return ["all powers"]

    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Topographic Features


def known_repeated_chars():
    """
    :param docs: the data
    :return: CSR Matrix with the num of repeation of each char in the list bellow
    """

    class InitTransformer(TransformerMixin, BaseEstimator):
        def __init__(self, char):
            self.char = char

        def fit(self, X, y=None):
            """All SciKit-Learn compatible transformers and classifiers have the
            same interface. `fit` always returns the same object."""
            return self

        def transform(self, X):
            def init_(post, char):
                return post.count(char) / len(post)

            X = [[init_(post, self.char)] for post in X]
            return csr_matrix(X)

        def get_feature_names(self):
            """Array mapping from feature integer indices to feature name"""
            return [self.char]

    feature_lst = []
    for char in [
        "!",
        "?",
        ".",
        "<",
        ">",
        "=",
        ")",
        "(",
        ":",
        "+",
        "*",
        "):",
        ":(",
        "(:",
        ":)",
    ]:
        feature_lst += [("KRC_" + char, InitTransformer(char))]

    return FeatureUnion(feature_lst)


# -----------------------------------------------------------------------------------------
# Repeated chars features

# return the normalized number of words with at least 3 repeated chars
def repeated_chars(data, language):
    # check for any repeated chars
    def in_word(word):
        for i in range(len(word) - 2):
            if word[i] == word[i + 1] == word[i + 2]:
                return True
        return False

    # check for the number of words with repeated chars
    def in_post(post):
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        repeated = 0
        for word in post:
            if in_word(word):
                repeated += 1
        return repeated / len(post)

    if data == "get feature names":
        return ["repeated chars"]

    # return the result
    return [[in_post(post)] for post in data]


def doubled_words(data, language):
    """
    :param data: the corpus
    :param language: the language of the data (ignored)
    :return: the num of doubled word normalized in the num f the words in the text
    """

    def in_post(post):
        num = 0
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        for i in range(len(post) - 1):
            if post[i] == post[i + 1]:
                num += 1
        return num / (len(post) - 1)

    if data == "get feature names":
        return ["doubled words"]

    return [[in_post(post)] for post in data]


def tripled_words(data, language):
    """
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: the num of tripled word normalized in the num of the words in the text
    """

    def in_post(post):
        num = 0
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        for i in range(len(post) - 2):
            if post[i] == post[i + 1] == post[i + 2]:
                num += 1
        return num / (len(post) - 2)

    if data == "get feature names":
        return ["tripled words"]

    return [[in_post(post)] for post in data]


def doubled_exclamation(data, language):
    """
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: the num of doubled ! normalized in the num of the letters in the text
    """
    if data == "get feature names":
        return ["doubled exclamation"]

    return [[post.count("!!") / len(post)] for post in data]


def tripled_exclamation(data, language):
    """
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: the num of doubled !! normalized in the num of the letters in the text
    """
    if data == "get feature names":
        return ["tripled exclamation"]

    return [[len(re.findall(r"!!!+", post)) / len(post)] for post in data]


def doubled_hyphen(data, language):
    """
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: the num of the words that contain at least 2 '-' normalized in the num of the words
    """

    def in_post(post):
        num = 0
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        for word in post:
            if word.count("-") >= 2:
                num += 1
        return num / len(post)

    if data == "get feature names":
        return ["doubled hyphen"]

    return [[in_post(post)] for post in data]


# -----------------------------------------------------------------------------------------
# General list
def general_list(data, lst):
    return [prevalence_rate(post, lst, True) for post in data]


# -----------------------------------------------------------------------------------------
# Language wealth


def words_wealth(data, language):
    """
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: list of the number of different types of words normalized in the number of words in the text
    """

    def in_post(post):
        post = re.findall(r"\b\w[\w-]*\b", post.lower())
        single_words = set(post)
        return len(single_words) / len(post)

    if data == "get feature names":
        return ["words wealth"]

    return [[in_post(post)] for post in data]


def n_word_in_post(post, num):
    vec = TfidfVectorizer(min_df=3, ngram_range=(1, 1)).fit([post])
    bag_of_words = vec.transform([post])
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    unique = [1 for tup in words_freq if tup[1] == num]
    post = re.findall(r"\b\w[\w-]*\b", post.lower())
    return len(unique) / len(post)


def once_words(data, language):
    """
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: the number of words that used only once in the post normalized in the number of the words
    """
    if data == "get feature names":
        return ["once words"]

    return [[n_word_in_post(post, 1)] for post in data]


def twice_words(data, language):
    """
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: the number of words that used only twice in the post normalized in the number of the words
    """
    if data == "get feature names":
        return ["twice words"]

    return [[n_word_in_post(post, 2)] for post in data]


def three_times_words(data, language):
    """
    :param language: the language of the data (ignored)
    :param data: the corpus
    :return: the number of words that used only three times in the post normalized in the number of the words
    """
    if data == "get feature names":
        return ["three times words"]

    return [[n_word_in_post(post, 2)] for post in data]
