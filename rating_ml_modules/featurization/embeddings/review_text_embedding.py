import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


class Tfidf:

    def __init__(self, n_gram_range: int = 2):

        self.max_ngram_range = n_gram_range
        self.vectorizer = TfidfVectorizer(analyzer="char",
                                          ngram_range=(2, self.max_ngram_range))

    def embed(self, text_dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        The embedding method computes a tfidf review text embedding for each row in the dataframe

        :param text_dataframe: List of review texts

        :return: Dataframe with spacy embedding columns and the review texts as indices
        """
        review_texts = list(text_dataframe.Review)
        tfidf_mat = self.vectorizer.fit_transform(review_texts)
        tfidf_df = pd.DataFrame.from_records(tfidf_mat.toarray()).applymap(lambda x: np.round(x, 4))
        tfidf_col_names = [f"dim_{i}" for i in range(tfidf_df.shape[1])]
        tfidf_df.columns = tfidf_col_names
        tfidf_df.index = review_texts
        tfidf_df.index.name = "text"
        return tfidf_df


class SpacyEmbedding:
    """
    This class computes the mean Roberta embedding for each Review
    """

    def __init__(self, spacy_model, max_len):
        self.spacy_model = spacy_model
        self.max_len = max_len
        self.embbedding_dim = len(self.spacy_model("the").vector)

    def mean_transformer_embedding(self, input_text):
        """
        This method computes the mean vector embedding accross all word vectors of an input text

        :param input_text: Text to be embedded

        :return: A text embedding numpy array vector
        """
        try:
            return np.mean(np.vstack([tkn.vector for tkn in self.spacy_model(input_text)][:self.max_len]), axis=0)
        except:
            return np.zeros(768, )

    def embed(self, text_dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        The embedding method computes a spacy review text embedding for each row in the dataframe

        :param text_dataframe: Dataframe containing a review column with texts to be embedded

        :return: Dataframe with spacy embedding columns and the review texts as indices
        """

        vector_dict_list = [
            {f"dim_{ind}": val for ind, val in enumerate(self.mean_transformer_embedding(input_text=txt))} for txt in
            tqdm(text_dataframe.Review, "Embed the Review Texts")]
        embedding_df = pd.DataFrame(vector_dict_list)
        embedding_df.index = text_dataframe.Review
        embedding_df.index.name = "text"
        return embedding_df
