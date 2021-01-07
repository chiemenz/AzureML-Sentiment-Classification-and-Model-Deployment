from typing import Dict, List
import numpy as np
import pandas as pd
import os


def get_polarity_dictionary(directory_path: str) -> Dict:
    """
    This method loads a sentiment polarity dictionary based on a root directory path

    :param directory_path: Root path to sentiment polarity dictionary

    :return: A dictionary with a token as a key and a polarity as a value either 0 if negative or 1 if positive
    """
    polarity_dict = {}
    for file in os.listdir(directory_path):
        if '.tsv' in file:
            f_path = os.path.join(directory_path, file)
            with open(f_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines()]
            for line in lines:
                split_line = line.split("\t")
                if len(split_line) > 1:
                    polarity_dict[split_line[0]] = float(split_line[1])

    return polarity_dict


class SentimentPolarity:
    """
    This class can be used to compute sentiment polarity score for a input text
    """

    def __init__(self, tokenizer, frequent_word_polarity_dict: Dict, adjective_polarity_dict: Dict):
        self.word_tokenizer = tokenizer
        self.freq_word_polarity = frequent_word_polarity_dict
        self.adj_polarity = adjective_polarity_dict

    def __call__(self, input_texts: List) -> pd.DataFrame:
        """
        Compute the min, max and mean polarity scores per document for a list of input texts

        :param input_texts: List of input texts for which the polarity scores should be computed

        :return: Dataframe which contains that polarity scores as columns and the text as an index
        """
        return pd.concat([self.text_2_polarity_scores(txt) for txt in input_texts], axis=0).set_index("text")

    def text_2_polarity_scores(self, input_text: str) -> pd.DataFrame:
        """
        This method computes the min, max, mean polarity scores for an input text both for adjectives and for
        frequent words based on polarity dictionaries

        :param input_text: Normalized input text for which the polarity scores should be computedd

        :return: Dictionary containing the min, max, mean polarity scores both for adjectives and frequent words as well
                 as the text for which these scores where computed
        """

        tokenized_text = self.word_tokenizer(input_text)
        adjective_scores = [0]
        freq_word_scores = [0]
        for token in tokenized_text:
            if token in self.freq_word_polarity.keys():
                freq_word_scores.append(self.freq_word_polarity[token])
            if token in self.adj_polarity.keys():
                adjective_scores.append(self.adj_polarity[token])

        polarity_df = pd.DataFrame(
            {'text': [input_text], 'min_adj': [np.min(adjective_scores)], 'max_adj': [np.max(adjective_scores)],
             'mean_adj': [np.mean(adjective_scores)], 'min_freq_w': [np.min(freq_word_scores)],
             'max_freq_w': [np.max(freq_word_scores)], 'mean_freq_w': [np.mean(freq_word_scores)]
             })

        return polarity_df
