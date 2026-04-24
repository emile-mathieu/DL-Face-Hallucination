import csv
import os

def save_metrics_to_csv(file_path, data, header=None):
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # write header only once
        if header and not file_exists:
            writer.writerow(header)

        writer.writerow(data)