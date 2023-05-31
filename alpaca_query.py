import logging
import wandb
from argparse import ArgumentParser
from src.evaluation import evaluate
from src.alpaca import get_alpaca_completions
from src.davinci import get_davinci_completions, get_davinci_completions_mp
from src.utils import load_dataset, load_json, parse_dataset_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def main(args):
    ds_name, question_type = parse_dataset_path(args)

    prompt_dict = load_json(args.prompt_path)
    prompt = prompt_dict[args.model_name][args.prompt_name]

    run_name = f"{args.model_name}_{ds_name}_{question_type}_{args.prompt_name}"
    logging.info(f"Started run {run_name}")

    wandb.init(project="temporal-ir", name=run_name)

    data = load_dataset(args.ds_path)
    print(data.head())

    if args.model_name == "alpaca-7b":
        predictions_df = get_alpaca_completions(
            args.model_name, data["Question"], run_name, prompt, args.batch_size
        )
    elif args.model_name == "text-davinci-003":
        predictions_df = get_davinci_completions_mp(
            args.model_name, data["Question"], run_name, prompt, args.batch_size
        )

    print("Received predictions... here is the head: ", predictions_df.head())

    evaluate(
        data,
        model_predictions=predictions_df,
        ds_name=ds_name,
        question_type=question_type,
    )


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--ds_path",
        required=True,
        help="Path to the output dir that will contain the logs and trained models",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        choices=["alpaca-7b", "stablelm-7b", "text-davinci-003"],
    )
    parser.add_argument("--prompt_name", required=True, type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument(
        "--prompt_path",
        required=True,
        help="Path to the prompt file",
        default="/home/wallat/temporal-llms/prompts.json",
    )

    args = parser.parse_args()
    main(args)
