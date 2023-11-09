# %%
# MIN_TRANSFORMERS_VERSION = '4.25.1'

# # check transformers version
# assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

# %%
from transformers import AutoTokenizer
import transformers
import torch


model_name = "tiiuae/falcon-7b-instruct"

generation_config = {"max_new_tokens": 32}
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=False)

# tokenizer.padding_side = "left"

pipe = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    batch_size=1,
    **generation_config,
)


out = pipe("Q: The capital of France is?\nA:")
print(out)

# %%
prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the question.\n\n### Input:\nI am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with 'Unknown'.\n\nQ: What is human life expectancy in the United States?\nA: 78 years\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower\n\nQ: Which party did he belong to?\nA: Republican Party\n\nQ: Where were the 1992 Olympics held?\nA: Barcelona, Spain.\n\nQ: {question}\n\n### Response:"

prompt = prompt.format(question="What is the capital of Germany?")

print(pipe(prompt))


