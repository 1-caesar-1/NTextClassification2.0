from gensim.sklearn_api import W2VTransformer
import numpy as np


class Word2VecTransfomer(W2VTransformer):
    def __init__(
        self,
        size=100,
        alpha=0.025,
        window=5,
        min_count=5,
        max_vocab_size=None,
        sample=1e-3,
        seed=1,
        workers=3,
        min_alpha=0.0001,
        sg=0,
        hs=0,
        negative=5,
        cbow_mean=1,
        hashfxn=hash,
        iter=5,
        null_word=0,
        trim_rule=None,
        sorted_vocab=1,
        batch_words=10000,
    ):
        super().__init__(
            size,
            alpha,
            window,
            min_count,
            max_vocab_size,
            sample,
            seed,
            workers,
            min_alpha,
            sg,
            hs,
            negative,
            cbow_mean,
            hashfxn,
            iter,
            null_word,
            trim_rule,
            sorted_vocab,
            batch_words,
        )

    def transform(self, docs):
        doc_vecs = []
        for doc in docs:
            ## for each document generate a word matrix
            word_vectors_per_doc = []
            for word in doc:
                ## handle out-of vocabulary words
                if word in self.gensim_model.wv:
                    word_vectors_per_doc.append(self.gensim_model.wv[word])

            word_vectors_per_doc = np.array(word_vectors_per_doc)
            ## take the column-wise mean of this matrix and store
            doc_vecs.append(word_vectors_per_doc.mean(axis=0))
        return np.array(doc_vecs)
