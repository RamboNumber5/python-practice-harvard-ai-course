import multiprocessing as mp
from datasets import load_dataset
from transformers import AutoTokenizer


def add_end_of_text(example):
    example['question'] = example['question'] + '<|endoftext|> '
    return example


def tokenize_function(examples):
    return tokenizer(examples["question"], truncation=True)

if __name__ == '__main__':
    dataset = load_dataset("squad")
    dataset = dataset.remove_columns(['id', 'context', 'answers', 'title'])
    dataset = dataset.map(add_end_of_text)
    model_checkpoint = "distilgpt2"
    # tokenizers are available in a python implementation of "Fast" implementation which uses the Rust language
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # Set the start method to 'spawn' which is the default for Windows
    mp.set_start_method('spawn', force=True)
    # By setting batched=True we process multiple elements of the dataset at once
    # num_proc sets the number of processes
    # Finally we remove the questions column because we won't need it now
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=["question"])
    print(tokenized_datasets)



