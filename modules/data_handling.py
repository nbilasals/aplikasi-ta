import os
import pandas as pd
from flask import request
from werkzeug.utils import secure_filename


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


def handle_uploaded_data():
    error_msg = None
    success_msg = None
    data_table = None
    dataset_size = (0, 0)
    columns = ["No", "Komentar", "Sentiment"]

    new_filename = "dataset_final.csv"
    save_location = os.path.join("database", new_filename)

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            error_msg = "No file selected."
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                file.save(save_location)
                success_msg = "File uploaded successfully."
            except Exception as e:
                error_msg = f"An error occurred while saving the file: {e}"
        else:
            error_msg = "Invalid file. Only CSV files are allowed."

    if os.path.exists(save_location):
        try:
            df = pd.read_csv(save_location, usecols=["Komentar", "Sentiment"])
            dataset_size = df.shape

            df = df[["Komentar", "Sentiment"]]
            df.insert(0, "No", range(1, len(df) + 1))

            table_html = """
                <div class="card">
                  <div class="table-responsive">
                    <table class="table">
                      <thead>
                        <tr>
                          <th>No</th>
                          <th>Komentar</th>
                          <th>Sentiment</th>
                        </tr>
                      </thead>
                      <tbody class="table-border">
            """
            for index, row in df.iterrows():
                table_html += f"""
                  <tr>
                    <td>{row['No']}</td>
                    <td>{row['Komentar']}</td>
                    <td>{row['Sentiment']}</td>
                  </tr>
                """
            table_html += """
                      </tbody>
                    </table>
                  </div>
                </div>
            """
            data_table = table_html
        except pd.errors.ParserError as e:
            error_msg = f"Error parsing CSV file: {e}"
        except ValueError as e:
            error_msg = f"Value error: {e}. Please ensure the file contains the required columns."
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"

    return columns, error_msg, success_msg, dataset_size, data_table
