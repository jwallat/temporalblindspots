# Temporal Blind Spots in LLMs

This repo contains experiments and necessary steps to reproduce the results from our WSDM '24 paper "Temporal Blind Spots in LLMs". 

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

We are using [TemporalQuestions]() and [ArchivalQA](https://github.com/WangJiexin/ArchivalQA). Currently, it is necessary to manually capitalize the "question" and "answer" the first line in the ArchivalQA files. I put these into a /data directory.
Additionally, we sample datasets from ArchivalQA. TODO: Description/code to set everything up.

Steps to prepare datasets: 
1. Create a data directory (e.g., /data)
2. Download Archival QA from here: https://github.com/WangJiexin/ArchivalQA
3. Download TemporalQuestions from here: https://www.dropbox.com/sh/fdepuisdce268za/AACtiPDaO_RwLCwhIwaET4Iba?dl=0
4. Download TempLAMA from here: https://github.com/google-research/language/tree/master/language/templama
5. Move the datasets to your data directory.



## Prompts
I used different options, which are specified in the prompts.json file. You can swap them there and specify the correct key as the ```--prompt_name``` argument. Default is "gpt_style" which we use in the experiments in the paper

## Models
The experiments use text-davinci-003 which requires you to register an API key at OpenAI and export it to OPENAI_API_KEY like 
```
export OPENAI_API_KEY=<key>
```

## Running Experiments
Running the experiments will mostly look like this:
```
python alpaca_query.py \
    --ds_path=/data/Event-focused Questions/Explicitly Time-Scoped Questions.csv \
    --model_name=alpaca-7b \
    --prompt_path=prompts.json \
    --prompt_name=gpt_style \
    --batch_size=$batch_size
```
The ```--ds_path``` is pointing toward the individual .csv files. ```--model_name``` might be "alpaca-7b" (more options in alpaca_query.py argparse arguments). ```--prompt_path``` should point toward a json file with prompts for the models. Per default, this is pointing at the prompts.json file. Lastly, ```--prompt_name``` is the key in the ```prompt_path``` file. 

Running this for all models will result in the big results table (Table 3).

## Results
Results will be written to standard output, a file with the model predictions will be saved, and logged to weights and biases (if set up).  

## Additional Experiments
Additional experiments will require additional steps. The scripts are in the `/scripts/` folder.

### Time Stratification
The file `/scripts/time_stratification.ipynb` covers the necessary steps. The configuration necessary is in the first cell. It is necessary to specify the dataset, prediction files, and interval information.  

### Time Referencing
For the time referencing experiments please have a look at `/scripts/absolute_relative_time_refs.ipynb` to create the datasets. 
If these datasets exist, just run the standard experiment setup and pass the corresponding datasets (that were generated in here).

### Temporal Errors
Temporal Errors can be computed via the `/scripts/temporal_error_checks.ipynb` file. Again, you need to pass the according dataset and predictions files in the second cell.  


## Can't find CUDA libdevice
I ran into this error a couple times. It seems that this was a common problem with the current tensorflow version. 
Check out the solution down here: https://www.tensorflow.org/install/pip
For me the CUDA path did not work out, so I needed to set CUDA_HOME: $CUDA_HOME=/home/wallat/.conda/envs/llm-inference/

## Citation
