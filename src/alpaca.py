import logging
import torch
import pandas as pd
from tqdm import tqdm
import torch
import transformers
from transformers import AutoTokenizer
from src.utils import load_json


# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler()],
# )


MODEL_REGISTRY = {
    # "llama-7b": "/home/idahl/models/llama_models_hf/llama-7b",
    # "llama-13b": "/home/idahl/models/llama_models_hf/llama-13b",
    "alpaca-7b": "/home/wallat/temporal-llms/models/alpaca_models_hf/alpaca-7b",
    "stablelm-7b": "OpenAssistant/stablelm-7b-sft-v7-epoch-3",
    "open-lama-7b": "VMware/open-llama-7b-v2-open-instruct",
    "open-lama-13b": "VMware/open-llama-13b-open-instruct",
    "falcon-7b": "tiiuae/falcon-7b-instruct",
    "falcon-40b": "tiiuae/falcon-40b-instruct",
    "mpt-7b": "mosaicml/mpt-7b-instruct",
    "redpajama-incite-7b": "togethercomputer/RedPajama-INCITE-7B-Instruct",
    "redpajama-incite-3b": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
}


def load_mpt(model_name, batch_size, generation_config):
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b",
        trust_remote_code=True,
        padding_side="left",
    )

    config = transformers.AutoConfig.from_pretrained(
        MODEL_REGISTRY[model_name], trust_remote_code=True
    )
    # config.max_seq_len = 4096  # (input + output) tokens can now be up to 4096

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_REGISTRY[model_name], config=config, trust_remote_code=True
    )

    pipe = build_pipe(model, tokenizer, batch_size, generation_config)
    # pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

    return pipe


def build_pipe(model, tokenizer, batch_size, generation_config):
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        torch_dtype=torch.bfloat16,
        batch_size=batch_size,
        trust_remote_code=True,
        **generation_config,
    )
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    return pipe


def load_redpajama(model_name, batch_size, generation_config):
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b",
        trust_remote_code=True,
        padding_side="left",
    )

    config = transformers.AutoConfig.from_pretrained(
        MODEL_REGISTRY[model_name], trust_remote_code=True
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_REGISTRY[model_name], config=config, trust_remote_code=True
    )

    pipe = build_pipe(model, tokenizer, batch_size, generation_config)
    return pipe


def load_generic(model_name, batch_size, generation_config):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REGISTRY[model_name],
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )

    pipe = build_pipe(
        MODEL_REGISTRY[model_name], tokenizer, batch_size, generation_config
    )
    return pipe


def load_model_pipeline(model_name, batch_size):
    logging.info(f"Loading model: {model_name}")

    generation_config = {"max_new_tokens": 32}

    if "mpt-" in model_name:
        pipe = load_mpt(model_name, batch_size, generation_config)
    elif "redpajama-" in model_name:
        pipe = load_redpajama(model_name, batch_size, generation_config)
    else:
        pipe = load_generic(model_name, batch_size, generation_config)

    return pipe


# def generate(pipe, prompt):
#     generation_config = {"max_new_tokens": 32}
#     generated_text = pipe(prompt, **generation_config)[0]["generated_text"]

#     return generated_text


# def make_model_request(pipe, question, prompt):
#     prompt = prompt.replace("{question}", question)

#     generated_text = generate(pipe, prompt)

#     # print("Generated text: ", generated_text)
#     split_response = generated_text.split("### Response:")
#     # print("Split: ", split_response)
#     response = split_response[-1].strip()
#     return response


def handle_retriever_text(row, args):
    relation_retriever_map = load_json(args.relation_retriever_map_path)

    retrieved_text = relation_retriever_map[row["Relation"]].replace(
        "<entity>", row["Entity"]
    )
    retrieved_text = retrieved_text.replace("<year>", str(row["Year"]))

    if args.retrieved_type == "hard":
        retrieved_text = retrieved_text.replace(
            "<retrieved_answer>", row["Retrieved_hard"]
        )
    elif args.retrieved_type == "soft":
        retrieved_text = retrieved_text.replace(
            "<retrieved_answer>", row["Retrieved_soft"]
        )

    return retrieved_text


def data_generator(data, prompt, args):
    for _, row in data.iterrows():
        input = prompt.replace("{question}", row["Question"])

        if args.retrieved_type != None:
            retrieved_text = handle_retriever_text(row, args)
            input = input.replace("{retrieved}", retrieved_text)

        print("\n\n\nInput: ", input)

        yield input


def get_alpaca_completions(model_name, data, run_name, prompt, args):
    job_df = {"q_id": [], "question": [], "answer": []}

    pipe = load_model_pipeline(model_name, args.batch_size)
    data_gen = data_generator(data, prompt, args)

    questions = data["Question"]

    print(questions.head())
    index = 0
    with tqdm(total=len(questions)) as t:
        for pipe_outs in pipe(data_gen):
            for out in pipe_outs:
                generated_text = out["generated_text"]
                question = questions[index]

                if model_name == "stablelm-7b":
                    split_response = generated_text.split("<|endoftext|><|assistant|>")[
                        1
                    ]
                    split_response = split_response.replace("<|assistant|>", "")
                    response = split_response.strip()
                elif "falcon" in model_name or "redpajama" in model_name:
                    print("Generated text: ", generated_text)
                    generation = generated_text.split(question)[1]
                    print("Generation: ", generation)
                    lines = generation.split(
                        "\n"
                    )  # Question in line 0, answer in line 1
                    response = lines[1].replace("A:", "").strip()
                    if "redpajama" in model_name:
                        if len(lines) > 2:
                            response = lines[2].strip()
                            if "Q:" in response:
                                response = ""
                        else:
                            response = lines[1].replace("A:", "").strip()

                    # split_response = generated_text.split("A:")
                    # response = split_response[-1].strip()
                    # response = response.split("\n")[0].strip()
                elif "mpt" in model_name or "open-lama" in model_name:
                    generation = generated_text.split("### Response:")
                    response = generation[1]
                    response = response.split("\n")[0].strip()
                else:
                    split_response = generated_text.split("### Response:")
                    response = split_response[-1].strip()

                job_df["q_id"].append(index)
                job_df["question"].append(question)

                if response == "":
                    job_df["answer"].append("<no response>")
                else:
                    job_df["answer"].append(response)

                final_df = pd.DataFrame(job_df)
                final_df.to_csv(f"{run_name}_predictions.csv", index=False, sep="\t")
                t.update(1)
                index += 1
    return final_df
