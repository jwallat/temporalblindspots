import wandb
from evaluate import load


def split_answers_with_multiple_options(data, question_type):
    dataset = []

    for index in range(0, len(data)):
        row = data.iloc[index]

        answers = row["Answer"]
        if "__or__" in answers:
            answers = answers.split("__or__")
        else:
            answers = [answers]

        dataset.append(
            {
                "id": str(index),
                "question": row["Question"],
                "answers": answers,
                "type": question_type,
            }
        )

    return dataset


def convert_to_references(dataset):
    # Convert to reference format (evaluate library)
    # {'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '0'}
    references = []

    for ele in dataset:
        answers = ele["answers"]
        ref = {
            "id": str(ele["id"]),
            "answers": {"answer_start": len(answers) * [0], "text": answers},
        }

        references.append(ref)

    return references


def convert_to_predictions(model_predictions):
    predictions = []

    for index in range(0, len(model_predictions)):
        row = model_predictions.iloc[index]

        predictions.append({"prediction_text": row["answer"], "id": str(row["q_id"])})

    # predictions[14]
    return predictions


def compute_contains_metric(references, predictions):
    # New metric "contains answer"
    num_contains = 0

    for pred, ref in zip(predictions, references):
        predicted_answer = pred["prediction_text"].lower()
        ref_answers = ref["answers"]["text"]
        ref_answers = [x.lower() for x in ref_answers]

        contained = False
        for ref_answer in ref_answers:
            if ref_answer in predicted_answer:
                contained = True

        if contained:
            num_contains += 1

    return (num_contains / len(references)) * 100


def compute_metrics(references, predictions):
    squad_metric = load("squad")
    results = squad_metric.compute(predictions=predictions, references=references)
    results["contains"] = compute_contains_metric(references, predictions)

    return results


def evaluate(data, model_predictions, question_type):
    dataset = split_answers_with_multiple_options(data, question_type)

    references = convert_to_references(dataset)
    predictions = convert_to_predictions(model_predictions)

    results = compute_metrics(references, predictions)
    wandb.log(results)
