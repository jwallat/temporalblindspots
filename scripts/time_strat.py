# %%
import sys

sys.path.insert(0, "../")
import pandas as pd
import re
from src.evaluation import (
    split_answers_with_multiple_options,
    convert_to_references,
    extract_questions,
    compute_metrics,
)
import matplotlib.pyplot as plt
import pandas as pd

# from src import *
# import src.utils as utils

# %% [markdown]
# # Focus time stratification

# %%
# Stratify questions based on year


question_type = "time"
# data, dataset_name = pd.read_csv('/home/wallat/temporal-llms/data/ArchivalQA/splits/ArchivalQATime/ArchivalQATime_train.csv'), "archivalQA_time_train"
# pred_data, model_name = pd.read_csv('/home/wallat/temporal-llms/data/predictions/alpaca-7b_ArchivalQA_time_train_set_gpt_style_predictions.csv', delimiter="\t"), "alpaca"
# pred_data, model_name = pd.read_csv('/home/wallat/temporal-llms/text-davinci-003_ArchivalQA_time_tuned_examples_predictions_2023_05_29T20:08.csv_ordered', delimiter="\t"), "davinci"


# upper_bound = 2008
# lower_bound = 1987
# intervals = [(year, year+1) for year in range(1987, 2008, 1)]

data, dataset_name = (
    pd.read_csv("/home/wallat/temporal-llms/data/templama/preprocessed/templamaQA.csv"),
    "TempLAMA",
)
pred_data, model_name = (
    pd.read_csv(
        "/home/wallat/temporal-llms/data/predictions/alpaca/datasets/alpaca-7b_TempLAMA_all_gpt_style_predictions.csv",
        delimiter="\t",
    ),
    "alpaca",
)
# pred_data, model_name = pd.read_csv('/home/wallat/temporal-llms/data/predictions/davinci/text-davinci-003_TempLAMA_all_tuned_examples_predictions_2023_05_30T09:17.csv_ordered', delimiter="\t"), "davinci"

upper_bound = 2021
lower_bound = 2009
intervals = [(year, year + 1) for year in range(2010, 2021, 1)]

questions = data["Question"].to_list()

data = data[:100]
pred_data = pred_data[:100]

data.head()

# %%
# Add years row to the original dataset


def data_add_years(data):
    years = []

    for index, row in data.iterrows():
        question = row["Question"]

        try:
            years.append(int(re.search(r"(\s\d{4})", question).group(1)))
        except:
            years.append(None)

    data = data.assign(Year=years)

    return data


# %%
# Add years row to the predictions

# pred_data = pd.read_csv('/home/wallat/temporal-llms/data/predictions/alpaca-7b_ArchivalQA_time_train_set_gpt_style_predictions.csv', delimiter="\t")
# pred_data.head()


def predictions_add_years(pred_data):
    years = []

    for index, row in pred_data.iterrows():
        question = row["question"]

        try:
            years.append(int(re.search(r"(\s\d{4})", question).group(1)))
        except:
            years.append(None)

    pred_data = pred_data.assign(Year=years)

    return pred_data, years


# %%
# # Find year numbers with regex
# import re

# years = []

# num_found = 0
# for question in questions:
#     try:
#         years.append(int(re.search(r"(\s\d{4})", question).group(1)))
#         num_found += 1
#     except:
#         # print(question)
#         ...

# print(f"Found {num_found} questions containing year numbers in the list of {len(questions)} questions")
# print("Minimum: ", min(years))
# print("Maximum: ", max(years))

# %%
# Filter for years in the specified boundaries


def filter_years_in_bounds(years, upper_bound, lower_bound):
    num = 0

    for year in years:
        if year > upper_bound:
            num += 1

    print(f"{num} years over upper bound {upper_bound}")

    num = 0

    for year in years:
        if year < lower_bound:
            num += 1

    print(f"{num} years under lower bound {lower_bound}")

    filtered_years = [year for year in years if year <= upper_bound]
    # len(filtered_years)

    filtered_years = [year for year in filtered_years if year >= lower_bound]
    print("Total years in dataset after filtering: ", len(filtered_years))

    return filtered_years


# %%
# Match IDs - original implementation uses q_id column, which does not work in this case
def convert_to_predictions(model_predictions):
    predictions = []

    for index in range(0, len(model_predictions)):
        row = model_predictions.iloc[index]

        predictions.append({"prediction_text": row["answer"], "id": str(index)})

    # predictions[14]
    return predictions


# %%
# Compute metrics for each interval


def compute_stratified_metrics(data, pred_data, intervals, model_name, dataset_name):
    results_list = []

    for interval in intervals:
        # for interval in intervals:
        lower, upper = interval

        # find all questions in the interval
        df_year_filtered = data[(data["Year"] >= lower) & (data["Year"] < upper)]
        # df_year_filtered

        df_pred_year_filtered = pred_data[
            (pred_data["Year"] >= lower) & (pred_data["Year"] < upper)
        ]
        # df_pred_year_filtered

        assert len(df_year_filtered) == len(df_pred_year_filtered)
        # break

        # Prepare dataset and compute metrics
        df_year_filtered = split_answers_with_multiple_options(
            df_year_filtered, question_type
        )
        references = convert_to_references(df_year_filtered)
        predictions = convert_to_predictions(df_pred_year_filtered)
        questions = extract_questions(df_year_filtered)

        if len(df_year_filtered) == 0:
            print(
                f"The performance in the for {len(df_year_filtered)} questions in {interval} is: undefined"
            )
            continue

        results = compute_metrics(references, predictions, questions)

        # for key in results.keys():
        #     wandb.log({f"{ds_name} {question_type} {key}": results[key]})
        # wandb.log(results)

        results_list.append(
            {
                "interval": str(interval),
                "EM": results["exact_match"],
                "F1": results["f1"],
                "Contains": results["contains"],
                "BEM": results["BEM_score"],
                "#questions": len(df_year_filtered),
            }
        )

        print(
            f"The performance in the for {len(df_year_filtered)} questions in {interval} is: {results}"
        )

    df = pd.DataFrame(results_list)
    df.head()
    df.to_csv(
        f"/home/wallat/temporal-llms/outputs/time_strat/metrics/{model_name}_{dataset_name}_metrics.csv",
        sep="\t",
    )

    return df


# %%
# df = pd.DataFrame(results_list)
# df.head()
# df.to_csv(f"/home/wallat/temporal-llms/outputs/time_strat/{model_name}_{dataset_name}_metrics.csv", sep='\t')

# %%

# import tikzplotlib

# def tikzplotlib_fix_ncols(obj):
#     """
#     workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
#     """
#     if hasattr(obj, "_ncols"):
#         obj._ncol = obj._ncols
#     for child in obj.get_children():
#         tikzplotlib_fix_ncols(child)


def save_metric_plots(results_df, dataset_name, model_name, intervals):
    for metric in ["EM", "F1", "Contains", "BEM"]:
        hist = results_df.plot(
            kind="bar",
            x="interval",
            y=metric,
            legend=False,
            ylabel=metric,
            xlabel="Year",
        )

        results_df["MA"] = results_df[metric].rolling(window=2).mean()
        plt.plot(results_df["MA"])
        plt.savefig(
            f"/home/wallat/temporal-llms/outputs/time_strat/{model_name}_{dataset_name}_intervals_{intervals[0][1]-intervals[0][0]}_years_{metric}.png"
        )

    # plt.show()


# tikzplotlib_fix_ncols(hist)
# tikzplotlib.save(f"/home/wallat/temporal-llms/outputs/time_strat/{model_name}_{dataset_name}_intervals_{intervals[0][1]-intervals[0][0]}_years_{metric}.tex")

# %%
# Predictions and models
models = [
    "alpaca",
    "davinci",
    "openllama",
    "falcon",
    "redpajama-3",
    "redpajama-7",
]  # , "alpaca", "davinci", "openllama", 'falcon', 'redpajama-3'

archival_predictions_paths_dict = {
    "alpaca": "/home/wallat/temporal-llms/data/predictions/alpaca/datasets/alpaca-7b_ArchivalQA_time_train_set_gpt_style_predictions.csv",
    "davinci": "/home/wallat/temporal-llms/data/predictions/davinci/datasets/text-davinci-003_ArchivalQA_time_tuned_examples_predictions_2023_05_29T20:08.csv_ordered",
    "openllama": "/home/wallat/temporal-llms/data/predictions/open-llama/7b/datasets/open-lama-7b_ArchivalQA_time_gpt_style_predictions.csv",
    "falcon": "/home/wallat/temporal-llms/data/predictions/falcon/7b/datasets/falcon-7b_ArchivalQA_time_tuned_examples_predictions.csv",
    "redpajama-3": "/home/wallat/temporal-llms/data/predictions/red-pajama/3b/datasets/redpajama-incite-3b_ArchivalQA_time_tuned_examples_predictions.csv",
    "redpajama-7": "/home/wallat/temporal-llms/data/predictions/red-pajama/7b/datasets/redpajama-incite-7b_ArchivalQA_time_tuned_examples_predictions.csv",
}

templama_predictions_paths_dict = {
    # "alpaca": "/home/wallat/temporal-llms/data/predictions/alpaca/datasets/alpaca-7b_TempLAMA_all_gpt_style_predictions.csv",
    # "davinci": "/home/wallat/temporal-llms/data/predictions/davinci/datasets/text-davinci-003_TempLAMA_all_tuned_examples_predictions_2023_05_30T09:17.csv_ordered",
    # "openllama": "/home/wallat/temporal-llms/data/predictions/open-llama/7b/datasets/open-lama-7b_TempLAMA_all_gpt_style_predictions.csv",
    # "falcon": "/home/wallat/temporal-llms/data/predictions/falcon/7b/datasets/falcon-7b_TempLAMA_all_tuned_examples_predictions.csv",
    "redpajama-7": "/home/wallat/temporal-llms/data/predictions/red-pajama/7b/datasets/redpajama-incite-7b_TempLAMA_all_tuned_examples_predictions.csv",
    "redpajama-3": "/home/wallat/temporal-llms/data/predictions/red-pajama/3b/datasets/redpajama-incite-3b_TempLAMA_all_tuned_examples_predictions.csv",
}

datasets = ["archivalQA_time_train"]

for dataset_name in datasets:
    if dataset_name == "archivalQA_time_train":
        data = pd.read_csv(
            "/home/wallat/temporal-llms/data/ArchivalQA/splits/ArchivalQATime/ArchivalQATime_train.csv"
        )
        pred_data = pd.read_csv(
            archival_predictions_paths_dict[model_name], delimiter="\t"
        )
        upper_bound = 2008
        lower_bound = 1987
        intervals = [(year, year + 1) for year in range(1987, 2008, 1)]
    elif dataset_name == "TempLAMA":
        data = pd.read_csv(
            "/home/wallat/temporal-llms/data/templama/preprocessed/templamaQA.csv"
        )
        pred_data = pd.read_csv(
            templama_predictions_paths_dict[model_name], delimiter="\t"
        )
        upper_bound = 2021
        lower_bound = 2009
        intervals = [(year, year + 1) for year in range(2010, 2021, 1)]

    for model_name in models:
        print(f"\n\n\nRunning {dataset_name} for {model_name}")

        if dataset_name == "archivalQA_time_train":
            pred_data = pd.read_csv(
                archival_predictions_paths_dict[model_name], delimiter="\t"
            )

        elif dataset_name == "TempLAMA":
            pred_data = pd.read_csv(
                templama_predictions_paths_dict[model_name], delimiter="\t"
            )

        questions = data["Question"].to_list()

        # TODO: Test run
        # data = data[:20]
        # pred_data = pred_data[:20]

        # print(pred_data['answer'])
        # break

        data = data_add_years(data)
        pred_data, years = predictions_add_years(pred_data)

        # filtered_years = filter_years_in_bounds(years, upper_bound, lower_bound)

        results = compute_stratified_metrics(
            data, pred_data, intervals, model_name, dataset_name
        )
        save_metric_plots(results, dataset_name, model_name, intervals)
