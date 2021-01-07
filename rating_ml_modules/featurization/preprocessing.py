import re
from typing import Dict

import pandas as pd


class Preprocessing:
    """
    This class is performing Review column text pre-processing
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        :param dataframe: A dataframe which contains a Review column with raw hotel review texts
        """
        self.dataframe = dataframe
        self.preprocessing_operations = []

    def remove_quotation_marks(self, quotation_mark_pattern: str = "('|´|`|\")"):
        """
        This method takes a quotation mark pattern and removes the quotation marks of the Review column values
        of a Dataframe

        :param quotation_mark_pattern: Regex pattern of quotation marks to be removed

        """
        self.dataframe['Review'] = self.dataframe.Review.apply(lambda x: re.sub(quotation_mark_pattern, "", x))
        self.preprocessing_operations += ['rmv_quotation_marks']

    def remove_special_characters(self,
                                  special_char_pattern: str = '(\x8e|î|æ|±|µ|_|\\|¾|\\||Â|â|é|\x94|«|\x8c|\x81|\x99|'
                                                              '\x95|\xa0|©|å|Â|ç|Ù|\x8a|«|¢|~|__Ç_)'):
        """
        This method removes special characters from the Review entries of the dataframe

        :param special_char_pattern: Pattern of special characters to be removed from the dataframe
        """

        self.dataframe['Review'] = self.dataframe.Review.apply(lambda x: re.sub(special_char_pattern, "", x))
        self.preprocessing_operations += ['rmv_special_chars']

    def comma_replace_remove_extra_whitespace(self):
        """
        This method replaces commas with whitespace and removes >1 whitespace with a single whitespace in the Review
        texts
        """

        self.dataframe['Review'] = self.dataframe.Review.apply(
            lambda x: re.sub(r"( )+", " ", re.sub("(,)", " ", x)).strip())
        self.preprocessing_operations += ['comma_2_whitespace_rmv_extra_whitespace']

    def __call__(self):
        self.remove_quotation_marks()
        self.remove_special_characters()
        self.comma_replace_remove_extra_whitespace()
        print(f"Preprocessing steps {self.preprocessing_operations}")
        return self.dataframe

    def __getitem__(self, item_name):
        if item_name == "dataframe":
            print(f"The following preprocessing operations have been performed {self.preprocessing_operations}")
        return getattr(self, item_name)


def relabel_reviews(review_dataframe: pd.DataFrame, rating_2_sentiment: Dict) -> pd.DataFrame:
    """
    This method performs relabels the review rating column into 3 buckets positive (4+5),
    neutral 3 and negative (1+2)

    :param review_dataframe: Dataframe containing a hotel review column and a corresponding rating column
    (1-5)
    :param rating_2_sentiment: Dictionary containing the mapping of the rating to a sentiment score

    :return: A Dataframe is returned which contains a normalized_rating column which just contains 3
    possible sentiment values negative (0), neutral (1) and positive (2)
    """

    review_dataframe['norm_rating'] = review_dataframe['Rating'].apply(lambda x: rating_2_sentiment[x])

    return review_dataframe
