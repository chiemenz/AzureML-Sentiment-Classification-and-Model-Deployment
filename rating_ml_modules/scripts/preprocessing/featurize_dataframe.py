"""
This script can be used to compute Features of hotel review Texts
the resulting DataFrame will be saved as a .csv file at ./data/datasets/preprocessed

Example arguments:
*your_absolute_filepath/tripadvisor_hotel_reviews.csv
--spacy-model
en_trf_robertabase_lg
--topic-number
30
"""
__author__ = "Christoph Hiemenz"
__version__ = "1.0"
__email__ = "ch314@gmx.de"

import argparse
import logging
import os

import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from paths import PATH_ADJ_POLARITY_FILES, PATH_FREQ_WORD_POLARITY_FILES, PATH_PREPROCESSED_DF
from rating_ml_modules.featurization.embeddings.review_text_embedding import SpacyEmbedding, Tfidf
from rating_ml_modules.featurization.preprocessing import Preprocessing, relabel_reviews
from rating_ml_modules.featurization.topic_modeling.topic_model import TopicModeling, topic_dictionary_2_dataframe
from rating_ml_modules.featurization.word_sentiment_polarity.sentiment_polarity import SentimentPolarity, \
    get_polarity_dictionary

english_stop_words = stopwords.words('english')


def main():
    parser = argparse.ArgumentParser(description='Script for extracting features from a hotel review dataset')
    parser.add_argument('dataset_path', type=str, help="Path to the  Kaggle Trip Advisor Reviews Dataset")
    embedding_args = parser.add_mutually_exclusive_group(required=True)
    embedding_args.add_argument('--spacy-model', type=str, default='en_trf_robertabase_lg',
                                help="Specify a spacy model for Text Vectorization")
    embedding_args.add_argument('--tfidf-character-ngram', type=int,
                                help="Specify a character n-gram >= 2 e.g. 2 means character bi-grams"
                                     "will be selected")
    parser.add_argument('--topic-number', type=int, default=30,
                        help="Number of topics for LDA model")

    arguments = parser.parse_args()

    # Load the Dataset
    dataset_path = os.path.normpath(arguments.dataset_path)
    review_df = pd.read_csv(dataset_path, encoding="utf-8")
    logger.info(f"Loaded Dataframe with shape {review_df.shape} from \n"
                f"path: {dataset_path}")

    # Relabel the DataFrame Rating to 3 bins 0=Negative (1+2) 1=Neutral (3) 2=Positive (4+5)
    rating_2_sentiment = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
    review_dataframe = relabel_reviews(review_dataframe=review_df, rating_2_sentiment=rating_2_sentiment)

    # Define Preprocessing Variables
    special_character_list = ['\x8e', 'î', 'æ', '±', 'µ', '_', '\\', '¾', '\|', 'Â', 'â', 'é', '\x94', '«', '\x8c',
                              '\x81', '\x99', '\x95', '\xa0', '©', 'å', 'Â', 'ç', 'Ù', '\x8a', '«', '¢', '~', '__Ç_']
    special_char_pattern = "(" + '|'.join(special_character_list) + ")"
    quotation_mark_pattern = "('|´|`|\")"

    # Perform preprocessing
    preprocessing_obj = Preprocessing(dataframe=review_dataframe)
    preprocessing_obj.remove_quotation_marks(quotation_mark_pattern=quotation_mark_pattern)
    preprocessing_obj.remove_special_characters(special_char_pattern=special_char_pattern)
    preprocessing_obj.comma_replace_remove_extra_whitespace()
    cleaned_dataframe = preprocessing_obj['dataframe']

    # Define a target label DataFrame
    target_label_df = cleaned_dataframe.loc[:, ['norm_rating']]
    target_label_df.index = cleaned_dataframe.Review
    target_label_df.index.name = "text"

    # Compute a text embedding
    selected_embedding = None
    embedded_df = None
    if not arguments.tfidf_character_ngram and arguments.spacy_model:
        spacy_model_name = arguments.spacy_model

        try:
            nlp_transformer = spacy.load(spacy_model_name)
        except Exception:
            raise (f"Please download the spacy model {spacy_model_name} via \n"
                   f"python -m spacy {arguments.spacy_model}")

        MAX_LEN = 200
        spacy_emb = SpacyEmbedding(spacy_model=nlp_transformer, max_len=MAX_LEN)
        embedded_df = spacy_emb.embed(text_dataframe=cleaned_dataframe).applymap(lambda x: np.round(x, 4))
        selected_embedding = spacy_model_name

    elif arguments.tfidf_character_ngram and arguments.tfidf_character_ngram >= 2:

        tfidf_emb = Tfidf(n_gram_range=arguments.tfidf_character_ngram)
        embedded_df = tfidf_emb.embed(text_dataframe=cleaned_dataframe).applymap(lambda x: np.round(x, 4))
        selected_embedding = f"tfidf_char_2to{arguments.tfidf_character_ngram}_embedding"

    logger.info(f"The Review texts were embedded via {selected_embedding} with dimensions {embedded_df.shape}")

    # Compute the sentiment polarity scores of the Review texts
    try:
        adjective_polarity_dictionary = get_polarity_dictionary(directory_path=PATH_ADJ_POLARITY_FILES)
        frequent_word_polarity_dictionary = get_polarity_dictionary(directory_path=PATH_FREQ_WORD_POLARITY_FILES)

        sentiment_polarity = SentimentPolarity(tokenizer=word_tokenize,
                                               frequent_word_polarity_dict=frequent_word_polarity_dictionary,
                                               adjective_polarity_dict=adjective_polarity_dictionary)

        review_texts = list(cleaned_dataframe.Review)
        sentiment_polarity_df = sentiment_polarity(review_texts)
        logger.info("Successfully computed sentiment polarity scores")
    except Exception:
        raise ImportError("Ensure to download the sentiment polarity lexica. See Readme.md")

    # Compute the Review length features
    cleaned_dataframe['review_len'] = cleaned_dataframe.Review.apply(lambda x: len(x.split(" ")))
    cleaned_dataframe['log_len'] = cleaned_dataframe.review_len.apply(lambda x: np.log(x))
    high_percentile = np.percentile(cleaned_dataframe.log_len, 95)
    low_percentile = np.percentile(cleaned_dataframe.log_len, 5)
    cleaned_dataframe['long_review'] = cleaned_dataframe['log_len'].apply(lambda x: 1 if x > high_percentile else 0)
    cleaned_dataframe['short_review'] = cleaned_dataframe['log_len'].apply(lambda x: 1 if x < low_percentile else 0)
    review_len_df = cleaned_dataframe.loc[:, ["long_review", "short_review", "norm_rating"]]
    review_len_df.index = cleaned_dataframe.Review
    review_len_df.index.name = "text"
    logger.info("Successfully computed review text length features")

    # Perform topic modeling to compute Topic embeddings for each Review text
    nltk.download('stopwords')
    try:
        nlp = spacy.load("en_core_web_md")
    except Exception:
        raise (f"Please download the spacy model en_core_web_md via \n"
               f"python -m spacy download en_core_web_md")

    topic_modeling = TopicModeling(list_of_texts=review_texts, tokenizer=word_tokenize, spacy_model=nlp,
                                   stop_words=english_stop_words)
    topic_modeling.preprocess()
    topic_modeling.token_list_2_lda_corpus()

    number_of_topics = arguments.topic_number
    if (type(number_of_topics) == int) and (number_of_topics > 1):
        chunk_size = 100
        topic_model_results = topic_modeling.fit_lda_topic_model(num_topics=number_of_topics,
                                                                 passes=20,
                                                                 chunksize=chunk_size)
        logger.info(f"Topic modeline was successfully completed")
        logger.info(f"LDA topic modeling coherence score: {np.round(topic_model_results['coherence_score'])}\n"
                    f"LDA topic modeling log perplexity: {np.round(topic_model_results['log_perplexity'], 3)}")

        review_topic_dictionary = topic_modeling(review_texts)
        topic_vec_df = topic_dictionary_2_dataframe(review_topic_dictionary=review_topic_dictionary)

    else:
        raise Exception("The number of topics needs to be and integer > 1")

    # Final DataFrame
    featurized_df = topic_vec_df.join(review_len_df,
                                      on="text", how='inner').join(sentiment_polarity_df,
                                                                   on="text", how="inner").join(embedded_df, on="text",
                                                                                                how="inner")
    logger.info(f"The preprocessed DataFrame has dimension {featurized_df.shape}")
    os.makedirs(PATH_PREPROCESSED_DF, exist_ok=True)
    save_path = os.path.join(PATH_PREPROCESSED_DF, f"{selected_embedding}preprocessed_df.csv")
    featurized_df.to_csv(save_path)
    logger.info(f"The preprocessed dataframe has been saved to {save_path}")


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    main()
