## Prompt templates

prompt_template = [
        "Requirement:\nChoose",
        "Requirement:\nConsidering the provided question and choices, identify the best answer's index",
        "Requirement:\nAnalyze the question to determine the most appropriate answer among the given choices",
        "Requirement:\nDetermine the ranking of the optimal answer for the next question"]

## Chain of thoughts

chain_of_thoughts_template = [
    "Which is a faster way to get home? Option 1: Take an 10 minutes bus, then an 40 minute bus, and finally a 10 minute train. Option 2: Take a 90 minutes train, then a 45 minute bike ride, and finally a 10 minute bus. Option 1 will take 10+40+10 = 60 minutes. Option 2 will take 90+45+10=145 minutes. Since Option 1 takes 60 minutes and Option 2 takes 145 minutes, Option 1 is faster.\n#### 1",


    "explain your reasoning step by step",


    "'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72'"]