from gensim.sklearn_api import D2VTransformer
import re


class Doc2VecTransfomer(D2VTransformer):
    """Tokenize input strings based on a simple word-boundary pattern."""

    def __init__(
        self,
        dm_mean=None,
        dm=1,
        dbow_words=0,
        dm_concat=0,
        dm_tag_count=1,
        docvecs=None,
        docvecs_mapfile=None,
        comment=None,
        trim_rule=None,
        size=100,
        alpha=0.025,
        window=5,
        min_count=5,
        max_vocab_size=None,
        sample=1e-3,
        seed=1,
        workers=3,
        min_alpha=0.0001,
        hs=0,
        negative=5,
        cbow_mean=1,
        hashfxn=hash,
        iter=5,
        sorted_vocab=1,
        batch_words=10000,
    ):
        super().__init__(
            dm_mean,
            dm,
            dbow_words,
            dm_concat,
            dm_tag_count,
            docvecs,
            docvecs_mapfile,
            comment,
            trim_rule,
            size,
            alpha,
            window,
            min_count,
            max_vocab_size,
            sample,
            seed,
            workers,
            min_alpha,
            hs,
            negative,
            cbow_mean,
            hashfxn,
            iter,
            sorted_vocab,
            batch_words,
        )

    def fit(self, X, y=None):
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        parser = lambda doc: token_pattern.findall(doc)
        d2v_texts = [parser(doc) for doc in X]
        super().fit(d2v_texts, y)
        return self

    def transform(self, X):
        ## split on word-boundary. A simple technique, yes, but mirrors what sklearn does to preprocess:
        ## https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/feature_extraction/text.py#L261-L266
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        parser = lambda doc: token_pattern.findall(doc)
        d2v_texts = [parser(doc) for doc in X]
        return super().transform(d2v_texts)
