from modules.preprocessing_utils import (
    case_folding,
    clean_text,
    replace_slang,
    word_tokenize_wrapper,
    stopwords_removal,
    get_stemmed_term,
    dict_slangs
)
from nltk.corpus import stopwords
import ast
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import ast

# so that matplot isn't error
matplotlib.use("Agg")


def convert_to_string(tokens):
    if isinstance(tokens, str):
        try:
            tokens = ast.literal_eval(tokens)
        except (ValueError, SyntaxError):
            tokens = tokens.split()
    return " ".join(tokens)


def preprocess_text(text):
    # Case folding
    text_lower = case_folding(text)

    # Cleaning
    text_cleaned = clean_text(text_lower)

    # Replace slang
    text_cleaned = replace_slang(text_cleaned, dict_slangs)

    # If the input text is tokenized, convert it back to string
    text_cleaned = convert_to_string(text_cleaned)

    # Tokenizing
    text_tokenized = word_tokenize_wrapper(text_cleaned)

    # Stopword removal
    text_stopwords_removed = stopwords_removal(text_tokenized)

    # Stemming
    text_stemmed = get_stemmed_term(text_stopwords_removed)

    # Convert tokens back to string after stemming
    text_stemmed_string = convert_to_string(text_stemmed)

    # Return the preprocessed text until stemming
    return text_stemmed_string


def generate_pie_chart(df):
    label_counts = df["Sentiment"].value_counts()

    # Plot the pie chart and add a legend with percentages
    plt.figure(figsize=(10, 8))  # Adjust the size of the pie chart (optional)

    # Customize the colors of the slices (optional)
    colors = ["#FACF32", "#FA6368", "#21CCAC"]

    # Plot the pie chart with custom colors and labels
    # Also, calculate the percentages for each category
    percentages = (label_counts / label_counts.sum()) * 100
    legend_labels = label_counts.index

    plt.pie(label_counts, colors=colors, startangle=90)

    # Add a title
    plt.title("Distribution of Sentiments")

    # Add the legend with percentages on the top left
    plt.legend(
        title="Label",
        loc="upper left",
        labels=[
            f"{label} ({percentage:.1f}% - {count})"
            for label, percentage, count in zip(
                label_counts.index, percentages, label_counts
            )
        ],
    )

    # Save the plot as a temporary image file
    image_path = os.path.join("static", "img/chart.png")
    plt.savefig(image_path, format="png", dpi=300)
    plt.close()

    return image_path


# convert the label
def convert(polarity):
    if polarity == "positive":
        return 0
    elif polarity == "neutral":
        return 1
    else:
        return 2


# term frequency - inverse document frequency
def calculate_tfidf(df_column, cvect=None, tfidf=None, max_features=1000):
    # Use provided vectorizers or create new ones
    if cvect is None:
        cvect = CountVectorizer(max_features=max_features)
        TF_vector = cvect.fit_transform(df_column)
    else:
        TF_vector = cvect.transform(df_column)

    # Normalize TF vector
    normalized_TF_vector = normalize(TF_vector, norm="l1", axis=1)

    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=max_features, smooth_idf=False)
        tfidf.fit(df_column)  # Fit the TfidfVectorizer with the data
    # No need to fit tfidf again, just use the idf values
    IDF_vector = tfidf.idf_

    # Calculate TF-IDF matrix
    tfidf_mat = normalized_TF_vector.multiply(IDF_vector).toarray()

    # Get feature names
    terms = tfidf.get_feature_names_out()

    # Sum TF-IDF frequency of each term through documents
    sums = tfidf_mat.sum(axis=0)

    # Calculate TF for each term
    term_freq = TF_vector.sum(axis=0).A1

    # Calculate IDF for each term
    idf_values = IDF_vector

    # Connect term to its sums frequency
    data = []
    for col, term in enumerate(terms):
        data.append((term, term_freq[col], idf_values[col], sums[col]))

    # Create a DataFrame with term statistics
    data_ranking = pd.DataFrame(data, columns=["term", "TF", "IDF", "rank"])
    data_ranking = data_ranking.sort_values("rank", ascending=False)

    return cvect, tfidf, IDF_vector, tfidf_mat, data_ranking


# for split
def convert_to_percent(decimal_value):
    percent_value = decimal_value * 100
    complementary_percent = 100 - percent_value
    result = f"{int(complementary_percent)}% | {int(percent_value)}%"
    return result


def sentiment_analysis_lexicon_indonesia(text, lexicon_positive, lexicon_negative):
    score = 0
    for word_pos in text:
        if word_pos in lexicon_positive:
            score += lexicon_positive[word_pos]
    for word_neg in text:
        if word_neg in lexicon_negative:
            score -= lexicon_negative[word_neg]
    polarity = 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
    return score, polarity

