import yaml
import os
import json
import time
import numpy as np
def read_yaml(path_to_yaml: str) -> dict:
    """
    Reads a yaml file and returns a dictionary
    """
    with open(path_to_yaml, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

def create_directory(dirs: list):
    """
    Creates a directory if it does not exist
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory is created at {dir_path}")

def save_local_df(data, data_path, index= False):
    """
    Saves a dataframe to a local file
    """
    data.to_csv(data_path, index=index)
    print(f"Dataframe is saved to {data_path}")

def save_reports(report: dict, report_path: str):
    """
    Saves a report to a local file
    """
    with open(report_path, 'w') as report_file:
        json.dump(report, report_file, indent=4)
    print(f"Report is saved to {report_path}")

def get_timestamp(name):
    """
    Returns a timestamp
    """
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name

def print_exams_average(myList):
    exams_avg = np.mean(myList, axis=0)
    return exams_avg