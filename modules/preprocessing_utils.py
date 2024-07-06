import nltk
import os
import pandas as pd
import pickle
import re
import string
from html import unescape
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

nltk.download("punkt")
nltk.download("stopwords")

# get indonesian stopwords
list_stopwords = stopwords.words("indonesian")

# add additional stopwords
list_stopwords.extend(["haaaalaaaah", "emangtidak", "bagaiaman", "ajah", "nih", "wts", "wtb", "nder", "eh", "ya", "sih", "deh", "dong", "sih", "tuh", "ya", "tapi", "yang", "kak", "min", "ah", "pls", "mjb", "bri", "spay", "ovo", "gopay", "bca",
                       "pakai", "bni", "dana", "shopeepay", "transfer", "saldo", "slot", "ga", "direct message", "whatsapp", "retweet", "gitu",
                       "rt", "jastip", "guys", "aku", "btw", "woi", "tweet", "inact", "ba", "bismillah", "halo", "hi", "hai", "follow", "dm", "wa", "thanks", "ya", "x", "likert", "kali", "receh", "bantu", "kasih", "help", "giveaway", "slot", "cuan", "only", "tba", "homescreenlockscreen"
                       ])

# add stopwords from txt file
txt_stopword = pd.read_csv(
    os.path.join("database", "stopwords.txt"), names=["stopwords"], header=None
)
list_stopwords.extend(txt_stopword["stopwords"][0].split(" "))

# To make it faster
combined_stopwords = set(list_stopwords)

# load stemmed term
with open(os.path.join("database", "term_dict.pkl"), "rb") as file:
    term_dict = pickle.load(file)

# case folding


def case_folding(text):
    return text.lower()

# cleaning


def clean_text(text):
    # Remove tab, new line, and back slice
    text = text.replace("\\t", " ").replace(
        "\\n", " ").replace("\\u", " ").replace("\\", "")

    # Remove non ASCII
    text = text.encode("ascii", "replace").decode("ascii")

    # Decode HTML entities
    text = unescape(text)

    # Remove mention, link, and hashtag
    # Update regex untuk hapus mention dengan karakter underscore
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'[@#][A-Za-z0-9]+|\w+:\/\/\S+', ' ',
                  text)  # Tetap hapus link dan hashtag

    # Remove incomplete URL
    text = text.replace("http://", " ").replace("https://", " ")

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Reduce word repetition to at most two consecutive characters
    text = re.sub(r'(.)\1+', r'\1', text)  # Mengurangi ke satu karakter saja

    # Remove superscript
    superscript_pattern = re.compile("["u"\U00002070"
                                     u"\U000000B9"
                                     u"\U000000B2-\U000000B3"
                                     u"\U00002074-\U00002079"
                                     u"\U0000207A-\U0000207E"
                                     u"\U0000200D"
                                     "]+", flags=re.UNICODE)
    text = superscript_pattern.sub('', text)

    # Remove laughter
    text = re.sub(r'\b(?:wkwk|haha|hehe|hihi|hoho|wkwkwk|ahahaha|hehehe|wkwkwkwkwk)\b',
                  '', text, flags=re.IGNORECASE)

    # Reduce word repetition
    text = re.sub(r'(.)\1+', r'\1\1', text)

    # Reduce repeated words
    text = re.sub(r'\b(\w+)(?:\W\1\b)+', r'\1', text, flags=re.IGNORECASE)

    # Remove single-character words
    text = re.sub(r'\b\w\b', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# slang word replacement


def replace_slang(text, slang_dict):
    words = TextBlob(text).words
    cleaned_text = ' '.join([slang_dict.get(word, word) for word in words])
    return cleaned_text


# load slang words
with open(os.path.join("database", "slang.txt"), "r", encoding="utf-8", errors='replace') as file:
    slangs = [line.strip().split(':') for line in file]
dict_slangs = {k.strip(): v.strip() for k, v in slangs}

# tokenization


def word_tokenize_wrapper(text):
    return word_tokenize(text)

# remove stopwords


def stopwords_removal(token_list):
    return [token for token in token_list if token.lower() not in combined_stopwords]

# apply stemming


def get_stemmed_term(document):
    stemmed_terms = [term_dict.get(term, None) for term in document]
    return [term for term in stemmed_terms if term is not None]
