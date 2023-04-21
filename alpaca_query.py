import logging
import wandb
from argparse import ArgumentParser
from src.evaluation import evaluate
from src.alpaca import get_alpaca_completions
from src.utils import load_dataset, load_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def main(args):
    if "Event-focused Questions" in args.ds_path:
        ds_name = "TemporalQuestions"
        if "Explicit" in args.ds_path:
            question_type = "explicit"
        else:
            question_type = "implicit"
    else:
        logging.error("Coundn't find dataset name, aborting")
        return

    prompt_dict = load_json(args.prompt_path)
    prompt = prompt_dict[args.model_name][args.prompt_name]

    run_name = f"{args.model_name}_{ds_name}_{question_type}_{args.prompt_name}"
    logging.info(f"Started run {run_name}")

    wandb.init(project="temporal-ir", name=run_name)

    data = load_dataset(args.ds_path)
    predictions_df = get_alpaca_completions(
        args.model_name, data["Question"], run_name, prompt
    )
    print(predictions_df.head())

    evaluate(data, model_predictions=predictions_df, question_type=question_type)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--ds_path",
        required=True,
        help="Path to the output dir that will contain the logs and trained models",
    )
    parser.add_argument("--model_name", required=True, type=str, choices=["alpaca-7b"])
    parser.add_argument("--prompt_name", required=True, type=str)
    parser.add_argument(
        "--prompt_path",
        required=True,
        help="Path to the prompt file",
        default="/home/wallat/temporal-llms/prompts.json",
    )

    args = parser.parse_args()
    main(args)
