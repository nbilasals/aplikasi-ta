import os
import pandas as pd
from flask import request
from werkzeug.utils import secure_filename

# Function to check if the uploaded file is a CSV


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

# Function to handle the uploaded data


def handle_uploaded_data():
    error_msg = None
    success_msg = None
    data_table = None
    dataset_size = (0, 0)
    columns = ["No", "Komentar", "Sentiment"]

    new_filename = "dataset_df.csv"
    save_location = os.path.join("database", new_filename)
    uploads_folder = os.path.join("database", "uploads")

    if request.method == "POST":
        file = request.files.get("file")
        print("Request method is POST")
        if not file:
            error_msg = "No file selected."
            print(error_msg)
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                # Save the uploaded file to the uploads folder
                file.save(os.path.join(uploads_folder, filename))
                print(f"File saved to {uploads_folder}/{filename}")

                # Read the uploaded CSV file
                new_data = pd.read_csv(os.path.join(uploads_folder, filename), usecols=[
                                       "Komentar", "Sentiment"])
                print("New data read successfully")

                # Append the new data to the existing dataset_final.csv
                if os.path.exists(save_location):
                    existing_data = pd.read_csv(save_location)
                    print("Existing data read successfully")
                    # Ensure column names match for concatenation
                    if set(existing_data.columns) == set(new_data.columns):
                        updated_data = pd.concat(
                            [existing_data, new_data], ignore_index=True)
                        updated_data.to_csv(save_location, index=False)
                        print("Data appended successfully")
                    else:
                        error_msg = "Column names in uploaded CSV do not match the existing dataset."
                        print(error_msg)
                else:
                    new_data.to_csv(save_location, index=False)
                    print("New data saved as the main dataset")

                success_msg = "File uploaded successfully and data appended."
                print(success_msg)
            except Exception as e:
                error_msg = f"An error occurred while processing the file: {e}"
                print(error_msg)
        else:
            error_msg = "Invalid file. Only CSV files are allowed."
            print(error_msg)
    else:
        print("Request method is not POST")

    if os.path.exists(save_location):
        df = pd.read_csv(save_location)
        dataset_size = df.shape

        df.insert(0, "No", range(1, len(df) + 1))
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
    else:
        print("Save location does not exist")
        return (
            columns,
            error_msg,
            success_msg,
            dataset_size,
            data_table,
            None,
            columns
        )
