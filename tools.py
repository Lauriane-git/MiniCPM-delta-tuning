from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import os

from dataset_import import dataset_import_function

# import the datasets


def import_datasets(random_shuffle_mmlu = False, random_shuffle_gsmk8 = False, num_samples = 200):

    # check whether the models have been imported, import them if not
    if not os.path.exists('/mmlu'):
        print("Datasets not found, importing them")
        dataset_import_function()

    dataset_mmlu = load_from_disk("mmlu/test")
    dataset_gsm8k = load_from_disk("gsm8k/test")
    
    if random_shuffle_mmlu:
        total_samples = len(dataset_mmlu)
        random_indices = random.sample(range(total_samples), num_samples)
        dataset_mmlu = dataset_mmlu.select(random_indices)
    else:
        dataset_mmlu = dataset_mmlu.select(range(num_samples))

    if random_shuffle_gsmk8:
        total_samples = len(dataset_gsm8k)
        random_indices = random.sample(range(total_samples), num_samples)
        dataset_gsm8k = dataset_gsm8k.select(random_indices)
    else:
        dataset_gsm8k = dataset_gsm8k.select(range(num_samples))

    return dataset_mmlu, dataset_gsm8k


# import MiniCPM chat model


def import_model():
    # the model has been downloaded locally from the Hugging Face model hub using  git clone
    # git@hf.co:openbmb/MiniCPM-2B-sft-bf16 -cf https://huggingface.co/docs/hub/models-downloading
    torch.manual_seed(0)

    # online mode
    path = 'openbmb/MiniCPM-2B-sft-bf16'
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)

    # local mode - model previously downloaded using git clone git@hf.co:openbmb/MiniCPM-2B-sft-bf16
    # path = "./MiniCPM-2B-sft-bf16"

    # model = AutoModelForCausalLM.from_pretrained(
    #     path,
    #     torch_dtype=torch.bfloat16,
    #     device_map="cuda",
    #     trust_remote_code=True,
        # local_files_only=True,
    # )

    tokenizer = AutoTokenizer.from_pretrained(path)

    return model, tokenizer


# check the model import and chat function


def test_model(model, tokenizer):
    responds, _ = model.chat(
        tokenizer,
        "山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？",
        temperature=0.8,
        top_p=0.8,
    )
    return responds

