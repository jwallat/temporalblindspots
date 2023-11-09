# Temporal IR & LLMs

This repo should cover all the experiments and necessary steps to reproduce them. 

## Conda Environment

```
conda create -n llm-inference 
conda activate llm-inference
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install black tqdm jupyter
pip install sentencepiece
pip install datasets
pip install git+https://github.com/huggingface/transformers
pip install chardet cchardet
pip install accelerate
pip install gradio
pip install -r requirements.txt

```

Furthermore, you might want to setup [Weights and Biases]().

## Datasets and Preprocessing

We are using [TemporalQuestions]() and [ArchivalQA](). Currently, it is necessary to manually capitalize the "question" and "answer" the first line in the ArchivalQA files. I put these into a /data directory.
Additionally, we sample datasets from ArchivalQA. TODO: Description/code to set everything up.

## Running Experiments
```
python alpaca_query.py \
    --ds_path=/data/Event-focused Questions/Explicitly Time-Scoped Questions.csv \
    --model_name=alpaca-7b \
    --prompt_path=prompts.json \
    --prompt_name=gpt_style \
    --batch_size=$batch_size
```
The ```--ds_path``` is pointing toward the individual .csv files. ```--model_name``` might be "alpaca-7b" (more options in alpaca_query.py argparse arguments). ```--prompt_path``` should point toward a json file with prompts for the models. Per default, this is pointing at the prompts.json file. Lastly, ```--prompt_name``` is the key in the ```prompt_path``` file. 

## Prompts
I used different options, which are specified in the prompts.json file. You can swap them there and specify the correct key as the ```--prompt_name``` argument

## Models
The experiments use text-davinci-003 which requires you to register an API key at OpenAI and export it to OPENAI_API_KEY

## Results
Results will be written to standard output, a file with the model predictions will be saved, and logged to weights and biases (if set up). Further analysis can be done with the scripts in the ```/scripts/``` folder. 

## Can't find CUDA libdevice
I ran into this error a couple times. It seems that this was a common problem with the current tensorflow version. 
Check out the solution down here: https://www.tensorflow.org/install/pip