import os

PATH_PACKAGE = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
# Directory containing the package
PATH_ROOT = os.path.normpath(os.path.join(PATH_PACKAGE, os.pardir))
# Path to sentiment polarity directory
PATH_POLARITY_DICTIONARIES = os.path.join(PATH_ROOT, "data", "polarity_data")
# Path to Adjective sentiment polarity .tsv file directory
PATH_ADJ_POLARITY_FILES = os.path.join(PATH_POLARITY_DICTIONARIES, "socialsent_hist_adj", "adjectives")
# Path to frequent word sentiment polarity .tsv file directory
PATH_FREQ_WORD_POLARITY_FILES = os.path.join(PATH_POLARITY_DICTIONARIES, "socialsent_hist_freq", "frequent_words")
# Preprocessed Dataset directory
PATH_PREPROCESSED_DF = os.path.join(PATH_ROOT, "data", "datasets", "preprocessed")
