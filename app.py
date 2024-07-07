from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
from modules import data_handling, preprocessing, analysis, evaluate
from modules.analysis_utils import sentiment_analysis_lexicon_indonesia, preprocess_text, calculate_tfidf
import pandas as pd
import pickle
from modules.evaluate_utils import generate_pie_chart_result, generate_wordcloud
from modules.evaluate import load_lexicons
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse


app = Flask(__name__)

# Load the logistic regression model and vectorizer
with open('database/classifier_model_lr.pkl', 'rb') as model_file:
    logistic_regression_model = pickle.load(model_file)

# Load the TF-IDF components
with open('database/vectorizer.pkl', 'rb') as f:
    cvect, tfidf, IDF_vector = pickle.load(f)
# Load lexicons
lexicon_positive, lexicon_negative = evaluate.load_lexicons()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analisis', methods=['GET', 'POST'])
def analisis():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!')
            return redirect(request.url)

        if file:
            # Save the file to a temporary location or read directly
            df = pd.read_csv(file)

            # Preprocess the text and transform with vectorizer
            df['Preprocessed_Text'] = df['Komentar'].apply(preprocess_text)
            # Calculate TF-IDF using previously trained vectorizer
            cvect, tfidf, IDF_vector, tfidf_mat, data_ranking = calculate_tfidf(
                df['Preprocessed_Text'])

           # Transform using TF-IDF
            X = cvect.transform(df['Preprocessed_Text'])
            X = X.multiply(IDF_vector)  # Apply IDF weights

            # Padding features if the number of features is less than expected
            if X.shape[1] < 1000:
                padding = np.zeros((X.shape[0], 1000 - X.shape[1]))
                X = np.hstack((X.toarray(), padding))

            # Perform sentiment analysis using Logistic Regression
            df['Logistic_Regression_Result'] = logistic_regression_model.predict(
                X)

            # Perform sentiment analysis using Lexicon-Based method
            df['Lexicon_Result'] = df['Preprocessed_Text'].apply(
                lambda text: sentiment_analysis_lexicon_indonesia(
                    text.split(), lexicon_positive, lexicon_negative)[1]
            )

            # Generate pie charts and wordclouds
           # Rename the columns for generating pie chart
            df_pred_lr = df.rename(
                columns={"Logistic_Regression_Result": "Predict_Result"})
            df_pred_lexicon = df.rename(
                columns={"Lexicon_Result": "Predict_Result"})

            # Generate pie charts and wordclouds
            chart_img_path_lr_predict = generate_pie_chart_result(
                df_pred_lr, "pie_chart_lr_predict.png")
            chart_img_path_lexicon_predict = generate_pie_chart_result(
                df_pred_lexicon, "pie_chart_lexicon_predict.png")

            df.rename(
                columns={"Preprocessed_Text": "Text_Token_Stop_Stem"}, inplace=True)
            # Generate dan save wordcloud images
            wordcloud_positive_lr_predict = generate_wordcloud(
                df, 'positive', 'wordcloud_positive_lr_predict.png', 'Logistic_Regression_Result')
            wordcloud_negative_lr_predict = generate_wordcloud(
                df, 'negative', 'wordcloud_negative_lr_predict.png', 'Logistic_Regression_Result')
            wordcloud_neutral_lr_predict = generate_wordcloud(
                df, 'neutral', 'wordcloud_neutral_lr_predict.png', 'Logistic_Regression_Result')

            wordcloud_positive_lexicon_predict = generate_wordcloud(
                df, 'positive', 'wordcloud_positive_lexicon_predict.png', 'Lexicon_Result')
            wordcloud_negative_lexicon_predict = generate_wordcloud(
                df, 'negative', 'wordcloud_negative_lexicon_predict.png', 'Lexicon_Result')
            wordcloud_neutral_lexicon_predict = generate_wordcloud(
                df, 'neutral', 'wordcloud_neutral_lexicon_predict.png', 'Lexicon_Result')

            # choose the column to show
            df_selected = df[
                [
                    "Komentar",
                    "Text_Token_Stop_Stem",
                    "Logistic_Regression_Result",
                    "Lexicon_Result"
                ]
            ]

            # Customize the DataFrame columns and index names
            columns = [
                "Komentar",
                "Preprocessed Text",
                "Logistic Regression",
                "Lexicon Based"
            ]
            df_selected.columns = columns

            # Add a new column 'No' with the desired index values
            df_selected.insert(0, "No", range(1, len(df_selected) + 1))

            # Set the 'No' column as the index
            df_selected.set_index("No")

            df = df_selected
            dataset_size = df_selected.shape

            # Render the results in a new page
            return render_template('analisis.html', data=df.to_html(index=False, classes="table table-striped"),
                                   chart_img_path_lr_predict=chart_img_path_lr_predict,
                                   chart_img_path_lexicon_predict=chart_img_path_lexicon_predict,
                                   wordcloud_positive_lr_predict=wordcloud_positive_lr_predict,
                                   wordcloud_negative_lr_predict=wordcloud_negative_lr_predict,
                                   wordcloud_neutral_lr_predict=wordcloud_neutral_lr_predict,
                                   wordcloud_positive_lexicon_predict=wordcloud_positive_lexicon_predict,
                                   wordcloud_negative_lexicon_predict=wordcloud_negative_lexicon_predict,
                                   wordcloud_neutral_lexicon_predict=wordcloud_neutral_lexicon_predict,
                                   dataset_size=dataset_size)

    return render_template('analisis.html')

# dataset


@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    (
        columns,
        error_msg,
        success_msg,
        dataset_size,
        data_table,
    ) = data_handling.handle_uploaded_data()

    return render_template(
        "dataset.html",
        columns=columns,
        error=error_msg,
        success=success_msg,
        dataset_size=dataset_size,
        data_table=data_table,
    )


@app.route("/training")
def training():
    data_preprocessed_head = preprocessing.preprocessed()

    (
        data_ranking_head,
        training_data,
        testing_data,
        split_amount,
    ) = analysis.analyze()
    
    return render_template(
        "training.html",
        data_preprocessed_head=data_preprocessed_head,
        data_ranking_head=data_ranking_head,
        training_data=training_data,
        testing_data=testing_data,
        split_amount=split_amount,
    )

# result


@app.route("/result", methods=["GET", "POST"])
def result():
    (
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
        wordcloud_neutral_lexicon
    ) = evaluate.evaluate_model_and_predict()

    return render_template(
        "result.html",
        acc_score_percentage_lr=acc_score_percentage_lr,
        precision_percentage_lr=precision_percentage_lr,
        recall_percentage_lr=recall_percentage_lr,
        f1_percentage_lr=f1_percentage_lr,
        chart_img_path_lr=chart_img_path_lr,
        chart_img_path_result_lr=chart_img_path_result_lr,
        acc_score_percentage_lex=acc_score_percentage_lex,
        precision_percentage_lex=precision_percentage_lex,
        recall_percentage_lex=recall_percentage_lex,
        f1_percentage_lex=f1_percentage_lex,
        chart_img_path_lex=chart_img_path_lex,
        chart_img_path_result_lexicon=chart_img_path_result_lexicon,
        error=error_msg,
        success=success_msg,
        dataset_size=dataset_size,
        data_table=data_table,
        wordcloud_positive_lr=wordcloud_positive_lr,
        wordcloud_negative_lr=wordcloud_negative_lr,
        wordcloud_neutral_lr=wordcloud_neutral_lr,
        wordcloud_positive_lexicon=wordcloud_positive_lexicon,
        wordcloud_negative_lexicon=wordcloud_negative_lexicon,
        wordcloud_neutral_lexicon=wordcloud_neutral_lexicon
    )


if __name__ == '__main__':
    app.run(debug=True, port=5001)
