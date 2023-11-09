# %%
import sys

sys.path.insert(0, "../")

from src.evaluation import *
import src.utils as utils

# import pandas as pd


# normal_data = utils.load_dataset("/home/wallat/temporal-llms/data/ArchivalQA/splits/ArchivalQATime/reference_types/sample/absolute_time_reference.csv")
# normal_predictions = utils.load_dataset("/home/wallat/temporal-llms/data/predictions/referencing_samples/alpaca-7b_ArchivalQA_abolute-time_sample_gpt_style_predictions.csv")

# no_time_data = utils.load_dataset("/home/wallat/temporal-llms/data/ArchivalQA/splits/ArchivalQATime/reference_types/sample/no_time_reference.csv")
# no_time_predictions = utils.load_dataset("/home/wallat/temporal-llms/alpaca-7b_ArchivalQA_no_time_sample_gpt_style_predictions.csv")

# normal_data.head()


# %% [markdown]
# ## Temporal Shift

# %% [markdown]
# A temporally-scoped question that will be answered (by a model) with pre/succeeding entity
#
# Test:
# Use the TempLAMA dataset and check for each year whether the model confuses the answer with the ground truth at year-1 & year+1

# %%
# Load the tempLAMA dataset and predictions
templama = utils.load_dataset(
    "/home/wallat/temporal-llms/data/templama/preprocessed/templamaQA.csv"
)
# alpaca_predictions = utils.load_dataset(
#     "/home/wallat/temporal-llms/data/predictions/alpaca/datasets/alpaca-7b_TempLAMA_all_gpt_style_predictions.csv"
# )
# alpaca_predictions = utils.load_dataset(
#     "/home/wallat/temporal-llms/data/predictions/davinci/datasets/text-davinci-003_TempLAMA_all_tuned_examples_predictions_2023_05_30T09:17.csv_ordered"
# )
alpaca_predictions = utils.load_dataset(
    "/home/wallat/temporal-llms/data/predictions/red-pajama/7b/datasets/redpajama-incite-7b_TempLAMA_all_tuned_examples_predictions.csv"
)

# %%
alpaca_predictions.head()

# %%
# for all questions where we have a +1/-1 year answer, check if the prediction is shifted

# do we only want to consider examples where the ground truth is different for the two years?

current_entity = ""

for _, row in alpaca_predictions.iterrows():
    id, question, answer = row
    print(question)
    print("GT answer: ", templama.iloc[id]["Answer"])
    break

# %%
from src.utils import load_json

template_map = load_json(
    "/home/wallat/temporal-llms/data/templama/relation_template_map.json"
)


# %%
def find_matching_template(question, template_map):
    for template in template_map.values():
        template = template.replace("<subject>", "")
        template = template.replace("<object>", "")
        template = template.replace("<year>?", "")
        if template in question:
            return template

    return None


# %%
plt = find_matching_template(question, template_map)
plt


# %%
def get_entity(question, template_map):
    template = find_matching_template(question, template_map)
    if template is None:
        return None

    entity = question.split(template)[0]
    return entity


# %%
entity = get_entity(question, template_map)
entity


# %%
def get_all_entity_rows(entity, data):
    return data[data["Question"].str.contains(entity)]


entity_rows = get_all_entity_rows("Tom Brady", templama)
entity_rows

# %%
from src.bem import predict as bem_predict
from src.evaluation import contains


bem, tokenizer, cls_id, sep_id = init_bem_model()

# def is_equivalent_bem(question, answer, prediction):
#     if "__or__" in answer:
#         answers = answer.split("__or__")
#         return is_equivalent_bem(question, answers[0], prediction) or is_equivalent_bem(question, answers[1], prediction)

#     return bem_predict(bem, question, answer, prediction, tokenizer, cls_id, sep_id) > 0.5


def is_equivalent_contains(question, answer, prediction):
    if "__or__" in answer:
        answers = answer.split("__or__")
        return is_equivalent_contains(
            question, answers[0], prediction
        ) or is_equivalent_contains(question, answers[1], prediction)

    return contains(answer, prediction)


res = is_equivalent_contains(
    "Tom Brady played for which team in 2010?", "Patriots", "New England Patriots"
)
res

# %%
# now find all examples where in year + 1 the answer is different

# for 2019 see if the given answer matches 2020 gt
# for 2020 see if the given answer matches 2019 gt
from tqdm import tqdm


def find_temporal_shift_places():
    current_entity = ""

    potential_shifts = 0
    actual_shifts = 0

    for line_id, row in tqdm(alpaca_predictions.iterrows(), total=50310):
        _, question, answer = row

        entity = get_entity(question, template_map)
        if entity == current_entity:
            continue

        current_entity = entity
        entity_rows = get_all_entity_rows(entity, templama)
        # print("\n\n\nFound new entity: ", entity)
        # print("This entity has ", len(entity_rows), " rows")
        # print(entity_rows)
        # print("\n\n")
        # if line_id > 100: break
        i = 0
        for i, obj in enumerate(entity_rows.iterrows()):
            q_id, gt_entity_row = obj
            # print(f"step {i} of {len(entity_rows)}")
            if i == len(entity_rows) - 1:
                continue

            cur_question = gt_entity_row["Question"]
            gt_answer = gt_entity_row["Answer"]
            next_gt_answer = entity_rows.iloc[i + 1]["Answer"]

            next_question = entity_rows.iloc[i + 1]["Question"]
            model_prediction = alpaca_predictions.iloc[q_id]["answer"]
            next_model_prediction = alpaca_predictions.iloc[q_id + 1]["answer"]
            # try:
            # except:
            #     print("Error at entity rows id", q_id)
            #     print(entity_rows.iloc[i + 1])
            #     continue

            # Identify temporal shift spots
            # TODO: be careful around answers containing __or__
            if gt_answer != next_gt_answer:
                gt_answers = gt_answer.split("__or__")
                next_gt_answers = next_gt_answer.split("__or__")
                # print("GT answers: ", gt_answers)
                # print("Next GT answers: ", next_gt_answers)

                # # Check forward: year, year + 1
                # print("********* Forward check *********")
                for gt_ans in gt_answers:
                    for next_gt_ans in next_gt_answers:
                        if gt_ans == next_gt_ans:
                            continue  # this is not a shift since both answers are the same
                        elif next_gt_ans not in gt_answers:
                            # print(f"\n\nFound potential shift spot if gt ({gt_ans}) is predicted as next gt ({next_gt_ans}) by model. Initial question was: {cur_question}")
                            # print(f"The actual model prediction is {model_prediction}")
                            # print(f"Shift if model predicted next GT")
                            # print(f"GT this year: {gt_ans}")
                            # print(f"GT next year: {next_gt_ans}")
                            # print(f"Model prediction this year: {model_prediction}")
                            potential_shifts += 1

                            if is_equivalent_contains(
                                cur_question, next_gt_ans, model_prediction
                            ):
                                # print("--------> Model prediction matches next gt answer - Found a shift!")
                                actual_shifts += 1

                # print("********* Backward check *********")
                # Check backward: year + 1, year
                for next_gt_ans in next_gt_answers:
                    for gt_ans in gt_answers:
                        if next_gt_ans == gt_ans:
                            continue
                        elif gt_ans not in next_gt_answers:
                            # print(f"\n\nFound potential shift spot if this gt ({next_gt_ans}) is predicted as previous gt ({gt_ans}) by model. Initial question was: {next_question}")
                            # print(f"The actual model prediction is {next_model_prediction}")

                            # print(f"Shift if model predicted previous GT")
                            # print(f"GT this year: {next_gt_ans}")
                            # print(f"GT previous year: {gt_ans}")
                            # print(f"Model prediction this year: {next_model_prediction}")
                            potential_shifts += 1

                            if is_equivalent_contains(
                                cur_question, gt_ans, next_model_prediction
                            ):
                                # print("--------> Next model prediction matches this gt answer - Found a shift!")
                                actual_shifts += 1

            # print("\n\n")

            # print("Found temporal shift spot for entity: ", entity)
            # print("Gt answer: ", gt_answer)
            # print("Next gt answer: ", next_gt_answer)

            # print("\nHere are the Model predictions: ")
            # print("Model prediction: ", model_prediction)
            # print("Next model prediction: ", next_model_prediction)

            # if is_equivalent_contains(question, next_gt_answer, model_prediction):
            #     print("Model prediction matches next gt answer - Found a shift!")
            #     actual_shifts += 1

            # if is_equivalent_contains(question, gt_answer, next_model_prediction):
            #     print("Next model prediction matches this gt answer - Found a shift!")
            #     actual_shifts += 1

            # # TODO: Since we look from both sides, this would be 2 potential shifts?
            # potential_shifts += 1

    print(f"Found {actual_shifts} of {potential_shifts} shifts")


# find_temporal_shift_places()

# %%


# %% [markdown]
# ## Time Invariance

# %% [markdown]
# The support a relation is so high that the time component is entirely disregarded.
#
# Test:
# Check if all answers over the 10 years in TempLAMA are always the same.

# %%
from tqdm import tqdm

bem, tokenizer, cls_id, sep_id = init_bem_model()


def find_time_invariant_examples():
    current_entity = ""

    num_entity_rows = 0
    time_invariant_examples = 0

    for line_id, row in tqdm(alpaca_predictions.iterrows(), total=50310):
        _, question, answer = row

        entity = get_entity(question, template_map)
        if entity == current_entity:
            continue

        current_entity = entity
        entity_rows = get_all_entity_rows(entity, templama)

        prediction = ""
        invariant = True
        one_correct = False

        for i, obj in enumerate(entity_rows.iterrows()):
            q_id, gt_entity_row = obj

            cur_prediction = alpaca_predictions.iloc[q_id]["answer"]

            if prediction == "":
                prediction = cur_prediction

            if is_equivalent_contains(
                gt_entity_row["Question"], prediction, cur_prediction
            ):
                one_correct = True

            if not is_equivalent_contains(
                gt_entity_row["Question"], prediction, cur_prediction
            ):
                invariant = False
                break

        if invariant and one_correct:
            time_invariant_examples += 1
        num_entity_rows += 1

    print(
        f"Found {time_invariant_examples} time invariant examples for {num_entity_rows} entities"
    )


find_time_invariant_examples()


# %%
def find_temporal_inertia():
    current_entity = ""
    entity_counter = 0

    potential_shifts = 0
    actual_shifts = 0

    for line_id, row in tqdm(alpaca_predictions.iterrows(), total=50310):
        _, question, answer = row

        entity = get_entity(question, template_map)
        if entity == current_entity:
            continue

        current_entity = entity
        entity_rows = get_all_entity_rows(entity, templama)
        entity_counter += 1
        # print("\n\n\nFound new entity: ", entity)
        # print("This entity has ", len(entity_rows), " rows")
        # print(entity_rows)
        # print("\n\n")
        # if line_id > 100: break
        shifts = []
        i = 0
        for i, obj in enumerate(entity_rows.iterrows()):
            q_id, gt_entity_row = obj
            # print(f"step {i} of {len(entity_rows)}")
            if i == len(entity_rows) - 1:
                continue

            cur_question = gt_entity_row["Question"]
            gt_answer = gt_entity_row["Answer"]
            next_gt_answer = entity_rows.iloc[i + 1]["Answer"]
            next_question = entity_rows.iloc[i + 1]["Question"]

            model_prediction = alpaca_predictions.iloc[q_id]["answer"]
            next_model_prediction = alpaca_predictions.iloc[q_id + 1]["answer"]

            # Identify temporal shift spots
            # TODO: be careful around answers containing __or__
            if gt_answer != next_gt_answer:
                gt_answers = gt_answer.split("__or__")
                next_gt_answers = next_gt_answer.split("__or__")
                # print("GT answers: ", gt_answers)
                # print("Next GT answers: ", next_gt_answers)

                # # Check forward: year, year + 1
                # print("********* Forward check *********")
                # for gt_ans in gt_answers:
                #     for next_gt_ans in next_gt_answers:
                #         if gt_ans == next_gt_ans:
                #             continue # this is not a shift since both answers are the same
                #         elif next_gt_ans not in gt_answers:
                #             # print(f"\n\nFound potential shift spot if gt ({gt_ans}) is predicted as next gt ({next_gt_ans}) by model. Initial question was: {cur_question}")
                #             # print(f"The actual model prediction is {model_prediction}")
                #             # print(f"Shift if model predicted next GT")
                #             # print(f"GT this year: {gt_ans}")
                #             # print(f"GT next year: {next_gt_ans}")
                #             # print(f"Model prediction this year: {model_prediction}")
                #             # potential_shifts += 1
                #             shift = {
                #                 "cur_question": cur_question,
                #                 "next_gt_ans": next_gt_ans,
                #                 "model_prediction": model_prediction
                #             }
                #             print("Shift: ", shift)
                #             shifts.append(shift)

                # print("********* Backward check *********")
                # Check backward: year + 1, year
                for next_gt_ans in next_gt_answers:
                    for gt_ans in gt_answers:
                        if next_gt_ans == gt_ans:
                            continue
                        elif gt_ans not in next_gt_answers:
                            # print(f"\n\nFound potential shift spot if this gt ({next_gt_ans}) is predicted as previous gt ({gt_ans}) by model. Initial question was: {next_question}")
                            # print(f"The actual model prediction is {next_model_prediction}")

                            # print(f"Shift if model predicted previous GT")
                            # print(f"GT this year: {next_gt_ans}")
                            # print(f"GT previous year: {gt_ans}")
                            # print(f"Model prediction this year: {next_model_prediction}")
                            # potential_shifts += 1
                            shift = {
                                "cur_question": cur_question,
                                "gt_ans": next_gt_ans,
                                "next_model_prediction": model_prediction,
                            }
                            # print("Shift: ", shift)
                            shifts.append(shift)
                            # is_equivalent_contains(cur_question, gt_ans, next_model_prediction):

        # Select last shift
        if len(shifts) > 0:
            potential_shifts += 1
            shift = shifts[-1]

            # After finding all, check for the last one
            if is_equivalent_contains(
                shift["cur_question"], shift["gt_ans"], shift["next_model_prediction"]
            ):
                # print("--------> Model prediction matches next gt answer - Found a shift!")
                print("Inertia:    ", shift)
                actual_shifts += 1

    print(f"Found {actual_shifts} of {potential_shifts} shifts")
    print("Entities in total: ", entity_counter)


# find_temporal_inertia()
