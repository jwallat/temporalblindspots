# Temporal IR & LLMs

This repo should cover all the experiments and necessary steps to reproduce them. 

## Conda Environment

LLM inference from max -> pull out the installed packages

Furthermore, you might want to setup [Weights and Biases]().

## Datasets and Preprocessing

We are using [TemporalQuestions]() and [ArchivalQA](). Currently, it is necessary to manually capitalize the "question" and "answer" the first line in the ArchivalQA files.

## Models
The experiments use text-davinci-003 which requires you to register an API key at OpenAI and export it to OPENAI_API_KEY
