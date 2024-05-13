from tqdm import tqdm

from gsm8k_tools import *
from mmlu_tools import *
from tools import *
from delta_tuning_tools import import_delta_model
from prompt_templates import chain_of_thoughts_template, prompt_template

import logging

logging.basicConfig(filename='output.log', level=logging.INFO)



################## Evaluate the baseline performance on the MMLU dataset - classification task ##################

def mmlu_performance_measure(model,
    tokenizer,
    dataset_mmlu,
    prompt_template="Requirement:\nChoose",
    cot=""):


    # template build. Standard template from https://github.com/OpenBMB/UltraEval/blob/main/datasets/mmlu/transform_gen_v1.py (original MiniCPM evaluation paper)
    instruction = " and respond with the letter of the correct answer, including the parentheses.\n"
    options = "Options:\n"
    answer_prompt = f"Answer:\n"

    i = 0
    count_good_answers = 0

    number_elements = len(dataset_mmlu)
    pbar = tqdm(range(number_elements))
    for element in dataset_mmlu:
        question = f"Question:\n{element['question']}\n"
        choices = ", ".join(
            [
                f"({chr(65 + idx)})  {choice}\n"
                for idx, choice in enumerate(element["choices"])
            ]
        )

        prompt = (
            cot + question + prompt_template + instruction + options + choices + answer_prompt
        )

        MiniCPM_responds, _ = model.chat(
            tokenizer,
            prompt,
            temperature=0.1,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id,
            # repetition_penalty = 1.02
        )

        correct_answer = f"({chr(65 + element['answer'])})"
        try:
            answer = extract_letter(MiniCPM_responds)
        except:
            answer = ""

        if answer == correct_answer:
            count_good_answers += 1

        i += 1
        pbar.update(1)
        if i > number_elements:
            break

    return count_good_answers/number_elements


################## Evaluate the baseline performance on the GSM8k dataset - generative task ##################

# 1k test set

def gsm8k_performance_measure(model,
    tokenizer,
    dataset_gsm8k,
    cot=""):
    # instance example: { 'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as
    # many clips in May. How many clips did Natalia sell altogether in April and May?', 'answer': 'Natalia sold 48/2
    # = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72', }
    
    correct_answers = 0
    i = 0
    
    number_elements = len(dataset_gsm8k)
    pbar = tqdm(range(number_elements))
    for element in dataset_gsm8k:

        MiniCPM_responds, _ = model.chat(
            tokenizer,
            cot + element['question'],
            temperature=0.1,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id,
            # repetition_penalty = 1.02
        )
        if is_correct(MiniCPM_responds, element):
            correct_answers += 1
        
        i += 1
        pbar.update(1)
        if i > number_elements:
            break

    return correct_answers/number_elements
    

##################                             Main                                          ##################

def main():
    number_elements = 250
    dataset_mmlu, dataset_gsm8k = import_datasets(random_shuffle_mmlu = True, random_shuffle_gsmk8 = True, num_samples = number_elements)
    model, tokenizer = import_model()

    # evaluate the baseline performance
    logging.info("Evaluating the baseline performance")
    mmlu_performance = mmlu_performance_measure(model, tokenizer, dataset_mmlu)
    logging.info(f"Model performance on the MMLU dataset: {mmlu_performance}")
    gsm8k_performance = gsm8k_performance_measure(model, tokenizer, dataset_gsm8k)
    logging.info(f"Model performance on the GSM8k dataset: {gsm8k_performance}")

    # influence of the prompt template on the performance
    
    logging.info("Evaluating the influence of the prompt template on the performance")
    for pt in prompt_template:
        logging.info(f"Prompt template: {pt}")
        mmlu_performance = mmlu_performance_measure(model, tokenizer, dataset_mmlu, prompt_template=pt)
        logging.info(f"Model performance on the MMLU dataset: {mmlu_performance}")
        
    # evaluate the performance with chain of thoughts prompt tuning
    logging.info("Evaluating the performance with chain of thoughts prompt tuning")
    for cot in chain_of_thoughts_template:
        logging.info(f"Chain of thoughts template: {cot}")
        mmlu_performance = mmlu_performance_measure(model, tokenizer, dataset_mmlu, cot=cot)
        logging.info(f"Model performance on the MMLU dataset: {mmlu_performance}")
        gsm8k_performance = gsm8k_performance_measure(model, tokenizer, dataset_gsm8k, cot=cot)
        logging.info(f"Model performance on the GSM8k dataset: {gsm8k_performance}")

    
    # evaluate the performance of the delta tuned model
    logging.info("Evaluating the performance of the delta tuned model")
    delta_model = import_delta_model(model) # importing the model modifies the backbone model, that then can be used for measuring the performance
    # the delta model variable itself is no use here
    mmlu_performance = mmlu_performance_measure(model, tokenizer, dataset_mmlu)
    logging.info(f"Delta tuned model performance on the MMLU dataset: {mmlu_performance}")
    gsm8k_performance = gsm8k_performance_measure(model, tokenizer, dataset_gsm8k)
    logging.info(f"Delta tuned model performance on the GSM8k dataset: {gsm8k_performance}")


if __name__ == "__main__":
    main()

