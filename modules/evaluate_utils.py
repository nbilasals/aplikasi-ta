from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pdfkit
from flask import render_template_string
import os
import ast
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
sns.set()


def plot_confusion_matrix(y_test, y_preds, classes, filename):
    cm = confusion_matrix(y_test, y_preds)
    cmat_df = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cmat_df, annot=True, cmap="Blues", fmt="d")
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    image_path = os.path.join("static", "img", filename)
    plt.savefig(image_path, format="png", dpi=300)
    plt.close()
    return image_path


def generate_wordcloud(df_pred, sentiment, filename, column_name):
    additional_stopwords = {'transaksi', 'bayar', 'pake', 'pakai', 'tunai', 'gue', 'singapura', 'malaysia', 'kredit', 'kartu',
                            'dbank', 'banget', 'surabaya', 'pro', 'tiket', 'ims', 'danamon', 'infinite', 'nya', 'ribu', 'kartu kredit'}
    stopwords = set(STOPWORDS).union(additional_stopwords)
   # Ensure the tokenized texts are valid lists before joining

    def convert_to_string(tokens):
        if isinstance(tokens, str):
            try:
                tokens = ast.literal_eval(tokens)
            except (ValueError, SyntaxError):
                tokens = tokens.split()
        return " ".join(tokens)

    df_pred['Text_Token_Stop_Stem'] = df_pred['Text_Token_Stop_Stem'].apply(
        convert_to_string)

    texts = " ".join(df_pred[df_pred[column_name] ==
                     sentiment]['Text_Token_Stop_Stem'])
    if not texts:
        texts = 'empty'  # Add a default word to avoid empty wordcloud issue

    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=stopwords, max_words=50).generate(texts)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    image_path = os.path.join("static", "img", filename)
    plt.savefig(image_path, format="png", dpi=300)
    plt.close()

    return image_path


def generate_pie_chart_result(df, filename):
    label_counts = df["Predict_Result"].value_counts()
    plt.figure(figsize=(8, 8))
    colors = ["#FACF32", "#FA6368", "#21CCAC"]
    percentages = (label_counts / label_counts.sum()) * 100
    legend_labels = label_counts.index
    plt.pie(label_counts, colors=colors, startangle=90)
    plt.title("Distribusi Sentimen")
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
    image_path = os.path.join("static", "img", filename)
    plt.savefig(image_path, format="png", dpi=300)
    plt.close()
    return image_path


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


def generate_bar_chart(sentiment_counts, output_path):
    # Calculate the total counts for each year
    sentiment_counts['total'] = sentiment_counts.sum(axis=1)

    # Define colors for each sentiment category
    colors = ["#FACF32", "#FA6368", "#21CCAC", "#FF9F00"]

    # Plot the bar chart with custom colors
    ax = sentiment_counts.plot(
        kind='bar', figsize=(10, 6), width=0.8, color=colors)

    plt.title('Jumlah Sentimen per Tahun')
    plt.xlabel('Tahun')
    plt.ylabel('Jumlah Komentar')
    plt.legend(title='Sentiment')
    plt.grid(True, axis='y')

    # Annotate each bar with its height
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x()
                    * 1.005, p.get_height() * 1.005))

    plt.savefig(output_path)
    plt.close()

    return output_path
