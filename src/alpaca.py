import logging
import torch
import pandas as pd
from tqdm import tqdm
import torch
import transformers


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


MODEL_REGISTRY = {
    # "llama-7b": "/home/idahl/models/llama_models_hf/llama-7b",
    # "llama-13b": "/home/idahl/models/llama_models_hf/llama-13b",
    "alpaca-7b": "/home/idahl/models/alpaca_models_hf/alpaca-7b",
}


def load_model_pipeline(model):
    logging.info(f"Loading model: {model}")

    pipe = transformers.pipeline(
        "text-generation",
        model=MODEL_REGISTRY[model],
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return pipe


def generate(pipe, prompt):
    generation_config = {"max_new_tokens": 32}
    generated_text = pipe(prompt, **generation_config)[0]["generated_text"]

    return generated_text


def make_model_request(pipe, question, prompt):
    prompt = prompt.replace("{question}", question)

    generated_text = generate(pipe, prompt)

    # print("Generated text: ", generated_text)
    split_response = generated_text.split("### Response:")
    # print("Split: ", split_response)
    response = split_response[-1].strip()
    return response


def data_generator(data, prompt):
    for question in data:
        input = prompt.replace("{question}", question)
        yield input


def get_alpaca_completions(model_name, data, run_name, prompt):
    job_df = {"q_id": [], "question": [], "answer": []}

    pipe = load_model_pipeline(model_name)

    data_gen = data_generator(data, prompt)

    print(data.head())
    for index in tqdm(range(0, len(data))):
        question = data[index]

        response = make_model_request(pipe, question, prompt)

        job_df["q_id"].append(index)
        job_df["question"].append(question)

        if response == "":
            job_df["answer"].append("<no response>")
        else:
            job_df["answer"].append(response)
        final_df = pd.DataFrame(job_df)
        final_df.to_csv(f"{run_name}_predictions.csv", index=False, sep="\t")
    return final_df
