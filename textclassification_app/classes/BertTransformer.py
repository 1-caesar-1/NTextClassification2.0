from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
import transformers as ppb
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class BertTransformer(BaseEstimator, TransformerMixin):
    """
    Adapter between BERT and the classic Scikit-Learn transformer interface
    Based on https://towardsdatascience.com/build-a-bert-sci-kit-transformer-59d60ddd54a5
    """

    def __init__(
            self,
            pretrained_weights='bert-base-uncased',
            bert_tokenizer=None,
            bert_model=None,
            embedding_func: Optional[Callable[[torch.tensor], torch.tensor]] = None
    ):
        """
        Build a transformer with a skleran interface for BERT
        :param pretrained_weights: Type of weights for the BERT. default: 'bert-base-uncased'
        :param bert_tokenizer: Instance of BERT tokenizer. If None, one will be created automatically
        :param bert_model: Instance of BERT model. If None, one will be created automatically
        :param embedding_func:
        """
        if bert_tokenizer is None:
            bert_tokenizer = ppb.BertTokenizer.from_pretrained(pretrained_weights)

        if bert_model is None:
            bert_model = ppb.BertModel.from_pretrained(pretrained_weights)

        if embedding_func is None:
            embedding_func = lambda x: x[0][:, 0, :].squeeze()

        self.tokenizer = bert_tokenizer
        self.model = bert_model
        self.model.eval()
        self.embedding_func = embedding_func

    def _tokenize(self, text: str) -> Tuple[torch.tensor, torch.tensor]:
        """
        Returns the tokens of the given line of text
        :param text: The text that needs to be converted to tokens
        :return: Tuple, list of tokens with mask and list of tokens without mask
        """
        # Tokenize the text with the provided tokenizer
        tokenized_text = self.tokenizer.encode(text, add_special_tokens=True)

        # Create an attention mask telling BERT to use all words
        attention_mask = [1] * len(tokenized_text)

        # bert takes in a batch so we need to unsqueeze the rows
        return (
            torch.tensor(tokenized_text).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def _tokenize_and_predict(self, text: str) -> torch.tensor:
        """
        Perform tokenize and prediction for the given text
        :param text: The text on which the work is to be performed
        :return: The text after embedding
        """
        tokenized, attention_mask = self._tokenize(text)

        embeddings = self.model(tokenized, attention_mask)
        return self.embedding_func(embeddings)

    def transform(self, text: List[str]):
        if isinstance(text, pd.Series):
            text = text.tolist()

        with torch.no_grad():
            return torch.stack([self._tokenize_and_predict(string) for string in text])

    def fit(self, X, y=None):
        """No fitting necessary so we just return ourselves"""
        return self


if __name__ == "__main__":
    df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv',
                     delimiter='\t', header=None)
    df = df[:2000]

    bert = BertTransformer()
    features = bert.fit_transform(df[0])

    labels = df[1]
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)
    print(lr_clf.score(test_features, test_labels))
