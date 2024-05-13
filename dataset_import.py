from datasets import load_dataset

def dataset_import_function():
    ## First dataset: MMLU
    # load the datasets from huggingface
    print("Loading datasets from huggingface")
    dataset_mmlu= load_dataset("cais/mmlu", 'all')
    print("MMLU dataset loaded")
    # save the datasets locally for further work
    dataset_mmlu.save_to_disk("mmlu")
    print("MMLU dataset saved")

    ## Second dataset: GSM8K
    # load the datasets from huggingface
    print("Loading datasets from huggingface")
    dataset_gsm8k = load_dataset("gsm8k", "main")
    print("GSM8K dataset loaded")
    # save the datasets locally for further work
    dataset_gsm8k.save_to_disk("gsm8k")
    print("GSM8K dataset saved")


if __name__ == "__main__":
    dataset_import_function()