import os
import pandas as pd
from flask import request, render_template
import pdfkit
from werkzeug.utils import secure_filename
from modules.file_utils import allowed_file
from modules.evaluate_utils import generate_pie_chart_result, generate_wordcloud, generate_bar_chart
from modules.preprocessing_utils import (
    case_folding,
    clean_text,
    word_tokenize_wrapper,
    stopwords_removal,
    get_stemmed_term
)


def report_display():
    error_msg = None
    success_msg = None
    data_table = None
    chart_report = None
    wordcloud_positive = None
    wordcloud_negative = None
    wordcloud_neutral = None
    bar_chart = None
    dataset_size = (0, 0)
    year = "all"

    filename = "dataset_final_report.csv"
    save_location_pred = os.path.join("database", filename)

    if request.method == "POST":
        year = request.form.get("year", "all")
        file = request.files.get("file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                file.save(save_location_pred)
                success_msg = "File uploaded successfully."
            except Exception as e:
                error_msg = f"An error occurred while saving the file: {e}"
        elif not file:
            error_msg = "No file selected."
        else:
            error_msg = "Invalid file. Only CSV files are allowed."
    else:
        year = request.form.get("year", "all")

    if os.path.exists(save_location_pred):
        df_pred = pd.read_csv(save_location_pred, delimiter=',')
        print("Initial DataFrame columns:", df_pred.columns)  # Debugging line

        # Extract year from month_year
        df_pred["year"] = df_pred["month_year"].apply(
            lambda x: int(x.split('-')[0]))

        df_pred["Text_Lower"] = df_pred["Komentar"].apply(case_folding)
        df_pred["Text_Cleaning"] = df_pred["Text_Lower"].apply(clean_text)
        df_pred["Text_Token"] = df_pred["Text_Cleaning"].apply(
            word_tokenize_wrapper)
        df_pred["Text_Token_Stop"] = df_pred["Text_Token"].apply(
            stopwords_removal)
        df_pred["Text_Token_Stop_Stem"] = df_pred["Text_Token_Stop"].apply(
            get_stemmed_term)
        df_pred["Text_String"] = df_pred["Text_Token_Stop_Stem"].astype(str)

        if year != "all":
            df_pred = df_pred[df_pred['year'] == int(year)]
        else:
            # Aggregate sentiment counts by year
            sentiment_counts = df_pred.groupby(
                ['year', 'Sentiment']).size().unstack(fill_value=0)
            bar_chart = generate_bar_chart(
                sentiment_counts, os.path.join("static", "img", "bar_chart_report.png"))

        # Generate pie chart results
        chart_report = generate_pie_chart_result(df_pred.rename(
            columns={"Sentiment": "Predict_Result"}), "chart_result_report.png")

        # Generate wordclouds
        wordcloud_positive = generate_wordcloud(
            df_pred, "positive", "wordcloud_positive.png", "Sentiment")
        wordcloud_negative = generate_wordcloud(
            df_pred, "negative", "wordcloud_negative.png", "Sentiment")
        wordcloud_neutral = generate_wordcloud(
            df_pred, "neutral", "wordcloud_neutral.png", "Sentiment")

        # Select columns to show
        df_pred_selected = df_pred[["Komentar", "Sentiment", "month_year"]]
        print("Selected DataFrame columns:",
              df_pred_selected.columns)  # Debugging line
        print("Selected DataFrame:", df_pred_selected)
        dataset_size = df_pred_selected.shape
        print(dataset_size)
        # Add a new column 'No' with the desired index values
        df_pred_selected.insert(0, "No", range(1, len(df_pred_selected) + 1))
        data_table = df_pred_selected.head(5).to_html(index=False)

    return (
        error_msg,
        success_msg,
        data_table,
        chart_report,
        dataset_size,
        wordcloud_positive,
        wordcloud_negative,
        wordcloud_neutral,
        bar_chart,
        year
    )


def print_report():
    filename = "dataset_final_report.csv"
    save_location_pred = os.path.join("database", filename)
    if os.path.exists(save_location_pred):
        df_pred = pd.read_csv(save_location_pred, delimiter=',')
        print("Initial DataFrame columns:", df_pred.columns)  # Debugging line

        # Extract year from month_year
        df_pred["year"] = df_pred["month_year"].apply(
            lambda x: int(x.split('-')[0]))

        df_pred["Text_Lower"] = df_pred["Komentar"].apply(case_folding)
        df_pred["Text_Cleaning"] = df_pred["Text_Lower"].apply(clean_text)
        df_pred["Text_Token"] = df_pred["Text_Cleaning"].apply(
            word_tokenize_wrapper)
        df_pred["Text_Token_Stop"] = df_pred["Text_Token"].apply(
            stopwords_removal)
        df_pred["Text_Token_Stop_Stem"] = df_pred["Text_Token_Stop"].apply(
            get_stemmed_term)
        df_pred["Text_String"] = df_pred["Text_Token_Stop_Stem"].astype(str)

        sentiment_counts = df_pred.groupby(
            ['year', 'Sentiment']).size().unstack(fill_value=0)
        sentiment_counts['total'] = sentiment_counts.sum(axis=1)
        sentiment_counts = sentiment_counts.reset_index()

        print("Sentiment Counts:", sentiment_counts)

        return sentiment_counts
