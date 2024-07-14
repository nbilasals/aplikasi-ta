import os
import pandas as pd
from nltk.corpus import stopwords

from modules.preprocessing_utils import (
    case_folding,
    clean_text,
    replace_slang,
    word_tokenize_wrapper,
    stopwords_removal,
    get_stemmed_term,
    dict_slangs
)


def preprocessed():
    # open the dataset
    filename = "dataset_df.csv"
    save_location = os.path.join("database", filename)
    df = pd.read_csv(save_location, delimiter=",")

    # case folding
    df["Text_Lower"] = df["Komentar"].apply(case_folding)

    # cleaning
    df["Text_Cleaning"] = df["Text_Lower"].apply(clean_text)

    # replace slang
    df["Text_Cleaning"] = df["Text_Cleaning"].apply(
        lambda x: replace_slang(x, dict_slangs))

    # tokenizing
    df["Text_Token"] = df["Text_Cleaning"].apply(word_tokenize_wrapper)

    # stopword removal
    df["Text_Token_Stop"] = df["Text_Token"].apply(stopwords_removal)

    # stemming
    df["Text_Token_Stop_Stem"] = df["Text_Token_Stop"].apply(get_stemmed_term)

    # save preprocessed dataset after preprocessing
    df.to_csv(os.path.join("database", "preprocessed_dataset.csv"), index=False)

    # choose the column to show
    df_selected = df[
        [
            "Komentar",
            "Text_Lower",
            "Text_Cleaning",
            "Text_Token",
            "Text_Token_Stop",
            "Text_Token_Stop_Stem",
            "Sentiment",
        ]
    ]

    # Customize the DataFrame columns and index names
    columns = [
        "Tweets",
        "Case Folding",
        "Cleaning",
        "Tokenization",
        "Stopwords Removal",
        "Stemming",
        "Label",
    ]
    df_selected.columns = columns

    # Add a new column 'No' with the desired index values
    df_selected.insert(0, "No", range(1, len(df_selected) + 1))

    # Set the 'No' column as the index
    df_selected.set_index("No")

    # choose the head of the document
    df_head = df_selected.head()

    # Convert the DataFrame to an HTML table
    data_preprocessed_head = df_head.to_html(
        index=False, classes="table table-striped")

    return data_preprocessed_head
