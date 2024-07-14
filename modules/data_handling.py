import os
import pandas as pd
from flask import request
from werkzeug.utils import secure_filename
from datetime import datetime


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


def handle_uploaded_data():
    error_msg = None
    success_msg = None
    data_table = None
    dataset_size = (0, 0)
    columns = ["No", "Komentar", "Sentiment"]

    initial_filename = "dataset_df.csv"
    save_directory = os.path.join("database", "uploads")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            error_msg = "No file selected."
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                # Read the uploaded CSV data
                df_new = pd.read_csv(file)

                # Ensure the new data has the correct columns
                if not set(columns[1:]).issubset(df_new.columns):
                    error_msg = "Invalid file structure. Ensure it has 'Komentar' and 'Sentiment' columns."
                    return columns, error_msg, success_msg, dataset_size, data_table, [], []

                # Filter the new dataframe to only include the required columns
                df_new = df_new[["Komentar", "Sentiment"]]

                # Load the initial dataset
                df_initial = pd.read_csv(
                    os.path.join("database", initial_filename))

                # Concatenate the initial dataset with the new data
                df_combined = pd.concat(
                    [df_initial, df_new], ignore_index=True)

                # Save the combined dataset to a new file
                new_filename = f"dataset_combined_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
                new_save_location = os.path.join(save_directory, new_filename)
                df_combined.to_csv(new_save_location, index=False)

                success_msg = "File uploaded and combined successfully."
                # Load the combined dataset for display
                df = df_combined

            except Exception as e:
                error_msg = f"An error occurred while processing the file: {e}"
        else:
            error_msg = "Invalid file. Only CSV files are allowed."
    else:
        # Load the most recent combined dataset if available
        combined_files = [f for f in os.listdir(
            save_directory) if allowed_file(f)]
        if combined_files:
            latest_combined_file = max(combined_files, key=lambda x: os.path.getctime(
                os.path.join(save_directory, x)))
            df = pd.read_csv(os.path.join(
                save_directory, latest_combined_file))
        else:
            df = pd.read_csv(os.path.join("database", initial_filename))

    if not error_msg and df is not None:
        dataset_size = df.shape

        df = df[["Komentar", "Sentiment"]]
        df.insert(0, "No", range(1, len(df) + 1))

        # Convert the DataFrame to an HTML table
        data_table = df.to_html(index=False, classes="table table-striped")

        data1 = df.to_dict(orient='records')
        columns1 = df.columns.values

        return (
            columns,
            error_msg,
            success_msg,
            dataset_size,
            data_table,
            data1,
            columns1,
        )

    return (
        columns,
        error_msg,
        success_msg,
        dataset_size,
        data_table,
        [],
        [],
    )


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


def handle_uploaded_data():
    error_msg = None
    success_msg = None
    data_table = None
    dataset_size = (0, 0)
    columns = ["No", "Komentar", "Sentiment"]

    initial_filename = "dataset_df.csv"
    save_directory = os.path.join("database", "uploads")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            error_msg = "No file selected."
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                # Read the uploaded CSV data
                df_new = pd.read_csv(file)

                # Ensure the new data has the correct columns
                if not set(columns[1:]).issubset(df_new.columns):
                    error_msg = "Invalid file structure. Ensure it has 'Komentar' and 'Sentiment' columns."
                    return columns, error_msg, success_msg, dataset_size, data_table, [], []

                # Filter the new dataframe to only include the required columns
                df_new = df_new[["Komentar", "Sentiment"]]

                # Load the initial dataset
                df_initial = pd.read_csv(
                    os.path.join("database", initial_filename))

                # Concatenate the initial dataset with the new data
                df_combined = pd.concat(
                    [df_initial, df_new], ignore_index=True)

                # Save the combined dataset to a new file
                new_filename = f"dataset_combined_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
                new_save_location = os.path.join(save_directory, new_filename)
                df_combined.to_csv(new_save_location, index=False)

                success_msg = "File uploaded and combined successfully."
                # Load the combined dataset for display
                df = df_combined

            except Exception as e:
                error_msg = f"An error occurred while processing the file: {e}"
        else:
            error_msg = "Invalid file. Only CSV files are allowed."
    else:
        # Load the most recent combined dataset if available
        combined_files = [f for f in os.listdir(
            save_directory) if allowed_file(f)]
        if combined_files:
            latest_combined_file = max(combined_files, key=lambda x: os.path.getctime(
                os.path.join(save_directory, x)))
            df = pd.read_csv(os.path.join(
                save_directory, latest_combined_file))
        else:
            df = pd.read_csv(os.path.join("database", initial_filename))

    if not error_msg and df is not None:
        dataset_size = df.shape

        df = df[["Komentar", "Sentiment"]]
        df.insert(0, "No", range(1, len(df) + 1))

        # Convert the DataFrame to an HTML table
        data_table = df.to_html(index=False, classes="table table-striped")

        data1 = df.to_dict(orient='records')
        columns1 = df.columns.values

        return (
            columns,
            error_msg,
            success_msg,
            dataset_size,
            data_table,
            data1,
            columns1,
        )

    return (
        columns,
        error_msg,
        success_msg,
        dataset_size,
        data_table,
        [],
        [],
    )
