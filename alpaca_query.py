import logging
import wandb
from argparse import ArgumentParser
from src.evaluation import evaluate
from src.alpaca import get_alpaca_completions
from src.davinci import get_davinci_completions_mp
from src.utils import load_dataset, load_json, parse_dataset_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def main(args):
    ds_name, question_type = parse_dataset_path(args)

    prompt_dict = load_json(args.prompt_path)

    if "open-lama" in args.model_name or "mpt" in args.model_name:
        prompt = prompt_dict["alpaca-7b"][args.prompt_name]
    elif "falcon" in args.model_name or "redpajama" in args.model_name:
        prompt = prompt_dict["text-davinci-003"][args.prompt_name]
    else:
        prompt = prompt_dict[args.model_name][args.prompt_name]

    run_name = f"{args.model_name}_{ds_name}_{question_type}_{args.prompt_name}"
    if args.retrieved_type is not None:
        run_name += f"_{args.retrieved_type}"

    logging.info(f"Started run {run_name}")

    wandb.init(project="temporal-ir", name=run_name)

    data = load_dataset(args.ds_path)
    print(data.head())

    if args.model_name == "text-davinci-003":
        predictions_df = get_davinci_completions_mp(
            args.model_name, data, run_name, prompt, args
        )
    else:
        # Assume huggingface model
        predictions_df = get_alpaca_completions(
            args.model_name, data, run_name, prompt, args
        )

    # predictions_df = load_dataset(
    #     "/home/wallat/temporal-llms/data/predictions/alpaca/datasets/alpaca-7b_TempLAMA_all_gpt_style_predictions.csv"
    # )

    print("Received predictions... here is the head: ", predictions_df.head())

    evaluate(
        data,
        model_predictions=predictions_df,
        ds_name=ds_name,
        question_type=question_type,
    )


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--ds_path", required=True)
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        choices=[
            "alpaca-7b",
            "stablelm-7b",
            "open-lama-7b",
            "open-lama-13b",
            "falcon-7b",
            "falcon-40b",
            "mpt-7b",
            "redpajama-incite-7b",
            "redpajama-incite-3b",
            "llama-2-7b-chat",
            "llama-2-13b-chat",
            "text-davinci-003",
        ],
    )
    parser.add_argument(
        "--retrieved_type",
        default=None,
        type=str,
        choices=["hard", "soft"],
    )
    parser.add_argument("--prompt_name", required=True, type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument(
        "--prompt_path",
        required=True,
        help="Path to the prompt file",
        default="/home/wallat/temporal-llms/prompts.json",
    )
    parser.add_argument(
        "--relation_retriever_map_path",
        help="Path to the relation maps to build the aritifically retrieved text",
        default="/home/wallat/temporal-llms/data/templama/relation_retriever_map.json",
    )

    args = parser.parse_args()
    main(args)
