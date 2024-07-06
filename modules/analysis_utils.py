import ast
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# so that matplot isn't error
matplotlib.use("Agg")


# Function to generate the pie chart and return it as an image
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
def calculate_tfidf(df_column, max_features=1000):
    # calc TF vector
    cvect = CountVectorizer(max_features=max_features)
    TF_vector = cvect.fit_transform(df_column)

    # normalize TF vector
    normalized_TF_vector = normalize(TF_vector, norm="l1", axis=1)

    # calc IDF
    tfidf = TfidfVectorizer(max_features=max_features, smooth_idf=False)
    tfidf.fit_transform(df_column)  # Fit the TfidfVectorizer with the data
    IDF_vector = tfidf.idf_

    # hitung TF x IDF sehingga dihasilkan TFIDF matrix / vector
    tfidf_mat = normalized_TF_vector.multiply(IDF_vector).toarray()

    # ranking the top term
    terms = tfidf.get_feature_names_out()

    # sum tfidf frequency of each term through documents
    sums = tfidf_mat.sum(axis=0)

    # calculating TF for each term
    term_freq = TF_vector.sum(axis=0).A1

    # calculating IDF for each term
    idf_values = IDF_vector

    # connecting term to its sums frequency
    data = []
    for col, term in enumerate(terms):
        data.append((term, term_freq[col], idf_values[col], sums[col]))

    data_ranking = pd.DataFrame(data, columns=["term", "TF", "IDF", "rank"])
    data_ranking = data_ranking.sort_values("rank", ascending=False)

    return cvect, tfidf, IDF_vector, tfidf_mat, data_ranking


# for split
def convert_to_percent(decimal_value):
    percent_value = decimal_value * 100
    complementary_percent = 100 - percent_value
    result = f"{int(complementary_percent)}% | {int(percent_value)}%"
    return result
