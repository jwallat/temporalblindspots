import pandas as pd
import json
import logging


def load_dataset(ds_path):
    try:
        data = pd.read_csv(ds_path)
        data["Question"]
    except:
        data = pd.read_csv(ds_path, delimiter="\t")

    return data


def load_json(path):
    with open(path) as file:
        json_data = json.load(file)

    return json_data


def parse_dataset_path(args):
    if "Event-focused Questions" in args.ds_path:
        ds_name = "TemporalQuestions"
        if "Explicit" in args.ds_path:
            question_type = "explicit"
        else:
            question_type = "implicit"
    elif "ArchivalQA" in args.ds_path:
        ds_name = "ArchivalQA"
        question_type = "all"
        if "Easy" in args.ds_path:
            question_type = "easy"
        elif "Hard" in args.ds_path:
            question_type = "hard"
        elif "NoTime" in args.ds_path:
            question_type = "notime"
        elif "Time" in args.ds_path:
            question_type = "time"
            if "reference_types/relative" in args.ds_path:
                if "2021" in args.ds_path:
                    question_type = "relative-time-2021"
                elif "2023" in args.ds_path:
                    question_type = "relative-time-2023"
                elif "random" in args.ds_path:
                    question_type = "relative-time-random"
                elif "off-by-3" in args.ds_path:
                    question_type = "relative-time-off-by-3"
                elif "off-by-5" in args.ds_path:
                    question_type = "relative-time-off-by-5"
                elif "off-by-7" in args.ds_path:
                    question_type = "relative-time-off-by-7"
                elif "off-by-10" in args.ds_path:
                    question_type = "relative-time-off-by-10"
                elif "off-by-15" in args.ds_path:
                    question_type = "relative-time-off-by-15"
                elif "off-by-20" in args.ds_path:
                    question_type = "relative-time-off-by-20"
            elif "reference_types/absolute" in args.ds_path:
                question_type = "abolute-time"
                if "random" in args.ds_path:
                    question_type = "absolute-time-random"
                elif "off-by-3" in args.ds_path:
                    question_type = "absolute-time-off-by-3"
                elif "off-by-5" in args.ds_path:
                    question_type = "absolute-time-off-by-5"
                elif "off-by-7" in args.ds_path:
                    question_type = "absolute-time-off-by-7"
                elif "off-by-10" in args.ds_path:
                    question_type = "absolute-time-off-by-10"
                elif "off-by-15" in args.ds_path:
                    question_type = "absolute-time-off-by-15"
                elif "off-by-20" in args.ds_path:
                    question_type = "absolute-time-off-by-20"
            elif "reference_types/sample/absolute" in args.ds_path:
                question_type = "abolute-time_sample"
                if "random" in args.ds_path:
                    question_type = "absolute-time-random_sample"
            if "reference_types/sample/relative" in args.ds_path:
                if "random" in args.ds_path:
                    question_type = "relative-time-random_sample"
                elif "2021" in args.ds_path:
                    question_type = "relative-time-2021_sample"

    elif "templama" in args.ds_path:
        ds_name = "TempLAMA"
        question_type = "all"
    else:
        logging.error("Coundn't find dataset name, aborting")
        raise Exception
    return ds_name, question_type
