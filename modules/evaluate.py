import os
import pandas as pd
import pickle
from flask import request
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import csv
from werkzeug.utils import secure_filename

from modules.preprocessing_utils import (
    case_folding,
    clean_text,
    word_tokenize_wrapper,
    stopwords_removal,
    get_stemmed_term
)
from modules.file_utils import allowed_file

from modules.evaluate_utils import plot_confusion_matrix, generate_pie_chart_result, sentiment_analysis_lexicon_indonesia, generate_wordcloud


def load_lexicons():
    lexicon_positive = {}
    lexicon_negative = {}
    with open('database/lexicon_positive.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            lexicon_positive[row[0]] = int(row[1])
    with open('database/lexicon_negative.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            lexicon_negative[row[0]] = int(row[1])
    return lexicon_positive, lexicon_negative


def evaluate_model_and_predict():
    filename = "data.pkl"
    save_location = os.path.join("database", filename)

    if os.path.exists(save_location):
        with open(save_location, "rb") as f:
            X_train, X_test, y_train, y_test, text_train, text_test = pickle.load(
                f)

        # Logistic Regression
        logModel = LogisticRegression(
            multi_class='multinomial', solver='lbfgs', max_iter=1000, penalty='l2', C=4)
        logModel.fit(X_train, y_train)
        y_preds_lr = logModel.predict(X_test)

        # Evaluation Metrics for Logistic Regression
        acc_score_lr = accuracy_score(y_test, y_preds_lr)
        precision_lr = precision_score(y_test, y_preds_lr, average="macro")
        recall_lr = recall_score(y_test, y_preds_lr, average="macro")
        f1_lr = f1_score(y_test, y_preds_lr, average="macro")

        acc_score_percentage_lr = f"{acc_score_lr * 100:.2f}%"
        precision_percentage_lr = f"{precision_lr * 100:.2f}%"
        recall_percentage_lr = f"{recall_lr * 100:.2f}%"
        f1_percentage_lr = f"{f1_lr * 100:.2f}%"

        classes = ["positive", "neutral", "negative"]
        # Plot confusion matrix for Logistic Regression
        chart_img_path_lr = plot_confusion_matrix(
            y_test, y_preds_lr, classes, "confmat_lr.png")

        filename = "classifier_model_lr.pkl"
        save_location = os.path.join("database", filename)
        with open(save_location, "wb") as f:
            pickle.dump(logModel, f)

        # Lexicon-Based
        lexicon_positive, lexicon_negative = load_lexicons()
        tokens = pd.Series(text_test).str.split()
        results = tokens.apply(lambda text: sentiment_analysis_lexicon_indonesia(
            text, lexicon_positive, lexicon_negative))
        _, y_preds_lexicon = zip(*results)

        # Evaluation Metrics for Lexicon-Based
        acc_score_lex = accuracy_score(y_test, y_preds_lexicon)
        precision_lex = precision_score(
            y_test, y_preds_lexicon, average="macro")
        recall_lex = recall_score(y_test, y_preds_lexicon, average="macro")
        f1_lex = f1_score(y_test, y_preds_lexicon, average="macro")

        acc_score_percentage_lex = f"{acc_score_lex * 100:.2f}%"
        precision_percentage_lex = f"{precision_lex * 100:.2f}%"
        recall_percentage_lex = f"{recall_lex * 100:.2f}%"
        f1_percentage_lex = f"{f1_lex * 100:.2f}%"

        # Plot confusion matrix for Lexicon-Based
        chart_img_path_lex = plot_confusion_matrix(
            y_test, y_preds_lexicon, classes, "confmat_lexicon.png")

    else:
        acc_score_percentage_lr = "None"
        precision_percentage_lr = "None"
        recall_percentage_lr = "None"
        f1_percentage_lr = "None"
        chart_img_path_lr = "None"

        acc_score_percentage_lex = "None"
        precision_percentage_lex = "None"
        recall_percentage_lex = "None"
        f1_percentage_lex = "None"
        chart_img_path_lex = "None"

    # Initialize variables outside conditional blocks
    chart_img_path_result_lr = "None"
    chart_img_path_result_lexicon = "None"

    error_msg = None
    success_msg = None
    data_table = None
    chart_img_path_result = None

    # filename = "new_dataset_qris.csv"
    filename = "dataset_df.csv"
    save_location_pred = os.path.join("database", filename)

    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            error_msg = "No file selected."
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            try:
                file.save(save_location_pred)
                success_msg = "File uploaded successfully."
            except Exception as e:
                error_msg = "An error occurred while saving the file."
        else:
            error_msg = "Invalid file. Only CSV files are allowed."

    if os.path.exists(save_location_pred):
        filename = "vectorizer.pkl"
        save_location_vec = os.path.join("database", filename)
        with open(save_location_vec, "rb") as f:
            cvect, tfidf, IDF_vector = pickle.load(f)

        filename = "classifier_model_lr.pkl"
        save_location_model = os.path.join("database", filename)
        with open(save_location_model, "rb") as f:
            logModel = pickle.load(f)

        df_pred = pd.read_csv(save_location_pred, delimiter=",")
        df_pred["Text_Lower"] = df_pred["Komentar"].apply(case_folding)
        df_pred["Text_Cleaning"] = df_pred["Text_Lower"].apply(clean_text)
        df_pred["Text_Token"] = df_pred["Text_Cleaning"].apply(
            word_tokenize_wrapper)
        df_pred["Text_Token_Stop"] = df_pred["Text_Token"].apply(
            stopwords_removal)
        df_pred["Text_Token_Stop_Stem"] = df_pred["Text_Token_Stop"].apply(
            get_stemmed_term)
        df_pred["Text_String"] = df_pred["Text_Token_Stop_Stem"].astype(str)

        TF_vector_new = cvect.transform(df_pred["Text_String"])
        normalized_TF_vector_new = normalize(TF_vector_new, norm="l1", axis=1)
        tfidf_mat_new = normalized_TF_vector_new.multiply(IDF_vector).toarray()

        predict_result_lr = logModel.predict(tfidf_mat_new)
        df_pred["Predict_Result_LR"] = predict_result_lr

        # Lexicon-Based Prediction
        lexicon_positive, lexicon_negative = load_lexicons()
        tokens_pred = df_pred["Komentar"].apply(word_tokenize_wrapper)
        results_pred = tokens_pred.apply(lambda text: sentiment_analysis_lexicon_indonesia(
            text, lexicon_positive, lexicon_negative))
        _, predict_result_lexicon = zip(*results_pred)
        df_pred["Predict_Result_Lexicon"] = predict_result_lexicon

        # Generate pie chart results for Logistic Regression
        chart_img_path_result_lr = generate_pie_chart_result(
            df_pred.rename(columns={"Predict_Result_LR": "Predict_Result"}), "chart_result_lr.png")

        # Generate pie chart results for Lexicon-Based
        chart_img_path_result_lexicon = generate_pie_chart_result(
            df_pred.rename(columns={"Predict_Result_Lexicon": "Predict_Result"}), "chart_result_lexicon.png")

        # Generate wordcloud
        wordcloud_positive_lr = generate_wordcloud(
            df_pred, "positive", "wordcloud_positive_logistic.png", "Predict_Result_LR")
        wordcloud_negative_lr = generate_wordcloud(
            df_pred, "negative", "wordcloud_negative_logistic.png", "Predict_Result_LR")
        wordcloud_neutral_lr = generate_wordcloud(
            df_pred, "neutral", "wordcloud_neutral_logistic.png", "Predict_Result_LR")

        wordcloud_positive_lexicon = generate_wordcloud(
            df_pred, "positive", "wordcloud_positive_lexicon.png", "Predict_Result_Lexicon")
        wordcloud_negative_lexicon = generate_wordcloud(
            df_pred, "negative", "wordcloud_negative_lexicon.png", "Predict_Result_Lexicon")
        wordcloud_neutral_lexicon = generate_wordcloud(
            df_pred, "neutral", "wordcloud_neutral_lexicon.png", "Predict_Result_Lexicon")
        # Select columns to show
        df_pred_selected = df_pred[
            [
                "Komentar",
                "Text_Token_Stop_Stem",
                "Predict_Result_LR",
                "Predict_Result_Lexicon"
            ]
        ]

        # Customize the DataFrame columns and index names
        columns_pred = [
            "Komentar",
            "Preprocessed Text",
            "Logistic Regression",
            "Lexicon-Based"
        ]
        df_pred_selected.columns = columns_pred

        # Saving the result
        filename = "predicted_dataset_qris.csv"
        save_location_pred = os.path.join("database", filename)
        df_pred_selected.to_csv(save_location_pred, index=False)

        dataset_size = df_pred_selected.shape

        # Add a new column 'No' with the desired index values
        df_pred_selected.insert(0, "No", range(1, len(df_pred_selected) + 1))

        # Set the 'No' column as the index
        df_pred_selected.set_index("No")

        # Take just the head
        df_pred_selected_head = df_pred_selected

        # Convert the DataFrame to an HTML table
        data_table = df_pred_selected_head.to_html(
            index=False, classes="table table-striped")

        data1 = df_pred_selected_head.to_dict(orient='records')
        columns1 = df_pred_selected_head.columns.values

    else:
        dataset_size = 0

    return (
        acc_score_percentage_lr,
        precision_percentage_lr,
        recall_percentage_lr,
        f1_percentage_lr,
        chart_img_path_lr,
        chart_img_path_result_lr,
        acc_score_percentage_lex,
        precision_percentage_lex,
        recall_percentage_lex,
        f1_percentage_lex,
        chart_img_path_lex,
        chart_img_path_result_lexicon,
        error_msg,
        success_msg,
        dataset_size,
        data_table,
        wordcloud_positive_lr,
        wordcloud_negative_lr,
        wordcloud_neutral_lr,
        wordcloud_positive_lexicon,
        wordcloud_negative_lexicon,
        wordcloud_neutral_lexicon,
        data1,
        columns1,
    )
