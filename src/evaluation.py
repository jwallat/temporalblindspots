import wandb
from evaluate import load
from src.bem import init_bem_model, predict
from tqdm import tqdm
import string


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


def extract_questions(dataset):
    questions = []

    for ele in dataset:
        questions.append(ele["question"])

    return questions


def convert_to_predictions(model_predictions):
    predictions = []

    for index in range(0, len(model_predictions)):
        row = model_predictions.iloc[index]

        predictions.append({"prediction_text": row["answer"], "id": str(row["q_id"])})

    # predictions[14]
    return predictions


def contains(ref, pred):
    ref = ref.lower()
    pred = pred.lower()

    ref = ref.translate(str.maketrans("", "", string.punctuation))
    pred = pred.translate(str.maketrans("", "", string.punctuation))

    return ref in pred


def compute_contains_metric(references, predictions):
    # New metric "contains answer"
    num_contains = 0
    scores = []

    for pred, ref in zip(predictions, references):
        try:
            predicted_answer = pred["prediction_text"].lower()
        except:
            # print(pred["prediction_text"])
            # Special character
            continue
        ref_answers = ref["answers"]["text"]
        ref_answers = [x.lower() for x in ref_answers]
        ref_answers = [
            x.translate(str.maketrans("", "", string.punctuation)) for x in ref_answers
        ]

        contained = False
        for ref_answer in ref_answers:
            if ref_answer in predicted_answer:
                contained = True

        if contained:
            num_contains += 1
            scores.append(1)
        else:
            scores.append(0)

    return (num_contains / len(references)) * 100, scores


def compute_bertscore(references, predictions):
    preds = []
    for pred in predictions:
        preds.append(pred["prediction_text"])

    refs = []
    for ref in references:
        refs.append(ref["answers"]["text"][0])

    bertscore_metric = load("bertscore")

    results = bertscore_metric.compute(
        predictions=preds, references=refs, lang="en", idf=True
    )

    p_bert = sum(results["precision"]) / len(results["precision"])
    r_bert = sum(results["recall"]) / len(results["recall"])
    f1_bert = sum(results["f1"]) / len(results["f1"])

    return p_bert, r_bert, f1_bert


def compute_bemscore(references, predictions, questions):
    # The default in the paper https://arxiv.org/pdf/2202.07654.pdf​ is 0.5 - Minimal improvement
    # has been observed when directly tuning the treshold to 0.56 on the training set.
    threshold = 0.5
    preds = []
    for pred in predictions:
        preds.append(pred["prediction_text"])

    refs = []
    for ref in references:
        refs.append(ref["answers"]["text"][0])

    bem, tokenizer, cls_id, sep_id = init_bem_model()

    scores = []
    for question, reference, prediction in tqdm(zip(questions, refs, preds)):
        score = predict(bem, question, reference, prediction, tokenizer, cls_id, sep_id)
        scores.append(1 if score > threshold else 0)

    bem_score = sum(scores) / len(scores)

    return bem_score


def compute_metrics(references, predictions, questions):
    squad_metric = load("squad")
    results = squad_metric.compute(predictions=predictions, references=references)
    results["contains"], _ = compute_contains_metric(references, predictions)

    # p_bert, r_bert, f1_bert = compute_bertscore(references, predictions)
    # results["Precision_bert"] = p_bert
    # results["Recall_bert"] = r_bert
    # results["F1_bert"] = f1_bert

    results["BEM_score"] = compute_bemscore(references, predictions, questions)

    return results


def evaluate(data, model_predictions, ds_name, question_type):
    dataset = split_answers_with_multiple_options(data, question_type)

    references = convert_to_references(dataset)
    predictions = convert_to_predictions(model_predictions)
    questions = extract_questions(dataset)

    results = compute_metrics(references, predictions, questions)

    for key in results.keys():
        wandb.log({f"{ds_name} {question_type} {key}": results[key]})
    wandb.log(results)
