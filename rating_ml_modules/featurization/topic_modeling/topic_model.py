import re
from typing import List, Dict

import gensim
import gensim.corpora as corpora
import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
from tqdm import tqdm


def topic_dictionary_2_dataframe(review_topic_dictionary):
    """
    This method takes a topic dictionary as an input and outputs a topic embedding dataframe

    :param review_topic_dictionary: dictionary with text as key and topic vector representation as value

    :return: DataFrame with Review texts as indices and topic vector dimensions as columns
    """
    df_list = []
    for key, topic_vector in tqdm(review_topic_dictionary.items(), total=len(review_topic_dictionary)):
        tmp_df = pd.DataFrame({'text': [key]})
        for ind, scalar in enumerate(topic_vector):
            tmp_df[f"topic_{ind}"] = scalar
        df_list.append(tmp_df)

    topic_vector_df = pd.concat(df_list, axis=0)
    topic_vector_df = topic_vector_df.set_index("text")
    topic_vector_df = topic_vector_df.applymap(lambda x: np.round(x, 4))
    return topic_vector_df


class TopicModeling:
    """
    This class performs topic modeling on a list of input texts and returns the topic model
    """

    def __init__(self, spacy_model, tokenizer, list_of_texts: List, stop_words: List,
                 punctuation_pattern: str = r"(\.|\?|:|_|,|!|$|%|&|-|=)",
                 pos_tag_include_list=["ADJ", "VERB", "ADV", "NOUNT"],
                 ):

        self.spacy_model = spacy_model
        self.stop_words = stop_words
        self.punctuation_pattern = punctuation_pattern
        self.keep_tags = pos_tag_include_list
        self.tokenizer = tokenizer
        self.texts = list_of_texts
        self.index2token_dictionary = {}
        self.tokenized_text = []
        self.lda_model = None
        self.method_list = [self.remove_punctuation,
                            self.remove_stopwords,
                            self.lemmatize_tokens]

    def normalize_input_text(self, text, *args):
        """
        This method applies a sequence of normalization steps on a text and returns a normalized text

        :param txt: Text to be normalized

        :return: Normalized Text
        """
        for fun in args:
            text = fun(text)
        return text

    def remove_punctuation(self, input_text: str):
        """
        This method removes the punctuation of an input text

        :param input_text: Text to be normalized

        :return: Text without punctuation
        """
        return re.sub(r"( )+", " ", re.sub(self.punctuation_pattern, " ", input_text)).strip()

    def remove_stopwords(self, input_text):
        """
        This method removes stopwords of an input text

        :param input_text: Text to be normalized

        :return: Text without stopwords
        """
        return ' '.join([tkn for tkn in input_text.split(" ") if tkn not in self.stop_words])

    def lemmatize_tokens(self, input_text):
        """
        This method lemmatizes an input text and keeps only a set of tokens corresponding to the
        defined keep_tag POS tag list

        :param input_text: Text to be normalized

        :return: Lemmatized text with POS tags of interest
        """
        return ' '.join([tkn.lemma_ for tkn in self.spacy_model(input_text) if tkn.pos_ in self.keep_tags])

    def preprocess_text(self, input_text: str):
        return self.normalize_input_text(input_text, *self.method_list)

    def preprocess(self):
        self.texts = [self.preprocess_text(txt) for txt in tqdm(self.texts)]

    def tokenize_text(self):
        """
        This method tokenizes the list of texts
        """
        self.tokenized_text = [self.tokenizer(txt) for txt in self.texts]

    def create_lda_model_dict(self):
        self.tokenize_text()
        self.index2token_dictionary = corpora.Dictionary(self.tokenized_text)

    def token_list_2_lda_corpus(self):
        if not self.index2token_dictionary:
            self.create_lda_model_dict()
        self.corpus = [self.index2token_dictionary.doc2bow(token_array) for
                       token_array in self.tokenized_text]

    def fit_lda_topic_model(self, num_topics: int, random_state: int = 42, chunksize: int = 100,
                            passes: int = 10, alpha: str = 'auto', **kwargs):
        if not self.corpus:
            self.token_list_2_lda_corpus()

        self.lda_model = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus,
            id2word=self.index2token_dictionary,
            num_topics=num_topics,
            random_state=random_state,
            chunksize=chunksize,
            passes=passes,
            alpha=alpha,
            **kwargs
        )

        return self.evaluate_lda_model()

    def evaluate_lda_model(self):

        perplexity = self.lda_model.log_perplexity(self.corpus)
        lda_coherence = CoherenceModel(model=self.lda_model,
                                       texts=self.tokenized_text,
                                       dictionary=self.index2token_dictionary,
                                       coherence='c_v')
        lda_coherence_score = lda_coherence.get_coherence()
        return {'log_perplexity': perplexity, 'coherence_score': lda_coherence_score}

    def get_topic_vector(self, input_text):

        if self.lda_model:
            text_indices = self.index2token_dictionary.doc2bow(self.tokenizer(input_text))

            topic_number = self.lda_model.num_topics
            topic_vec = np.zeros(topic_number)
            for topic in self.lda_model[text_indices]:
                topic_vec[topic[0]] = topic[1]
            return topic_vec
        else:
            raise Exception("A topic model needs to be trained to get a topic vector")

    def __call__(self, input_texts: List) -> Dict:
        if self.lda_model:
            clean_texts = [self.preprocess_text(txt) for txt in tqdm(input_texts)]
            return {txt: self.get_topic_vector(indices) for txt, indices in zip(input_texts, clean_texts)}

        else:
            raise Exception("A topic model has to be fitted before this method can be called")

    def __getitem__(self, item_name):
        return getattr(self, item_name)
