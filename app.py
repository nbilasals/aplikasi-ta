from flask import Flask, render_template, request
from modules import data_handling, preprocessing, analysis, evaluate

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


# @app.route('/upload', methods=['POST'])
# def upload():
#     columns, error_msg, success_msg, dataset_size, data_table = handle_uploaded_data()
#     # Get the dataset table from data_handling.py
#     dataset_table = get_dataset_table()
#     return render_template('dataset.html', columns=columns, error_msg=error_msg,
#                            success_msg=success_msg, dataset_size=dataset_size,
#                            data_table=data_table, dataset_table=dataset_table)


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


@app.route('/analisis')
def analisis():
    return render_template('analisis.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
