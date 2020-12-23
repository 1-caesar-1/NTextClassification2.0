from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer


class BertTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            bert_tokenizer=None,
            bert_model=None,
            embedding_func: Optional[Callable[[torch.tensor], torch.tensor]] = None,
    ):
        if not bert_model:
            bert_model = BertModel.from_pretrained("bert-base-uncased")
        if not bert_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.tokenizer = bert_tokenizer
        self.model = bert_model
        self.model.eval()
        self.embedding_func = embedding_func

        if self.embedding_func is None:
            self.embedding_func = lambda x: x[0][:, 0, :].squeeze()

    def _tokenize(self, text: str) -> Tuple[torch.tensor, torch.tensor]:
        # Tokenize the text with the provided tokenizer
        tokenized_text = self.tokenizer.encode_plus(text,
                                                    add_special_tokens=True,
                                                    truncation=True
                                                    )["input_ids"]

        # Create an attention mask telling BERT to use all words
        attention_mask = [1] * len(tokenized_text)

        # bert takes in a batch so we need to unsqueeze the rows
        return (
            torch.tensor(tokenized_text).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def _tokenize_and_predict(self, text: str) -> torch.tensor:
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
