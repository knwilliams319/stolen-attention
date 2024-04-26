#########################################################################################################################################
# OpenbookQA comes with raw data for the train, dev, and test sets in jsonl format. It also has supporting facts in a .txt format.
# This script uses that raw data to make processed (Tensor) variants of each question that are ready to be used in a DataLoader. 

# USAGE: Call this script from the command line to call `preprocess_raw` and generate the information necessary to efficiently create 
#        distractor prompts that include random facts. Then `create_questions` will be called to generate tensors representing the 
#        distracting questions for the train, dev and test sets. 
#########################################################################################################################################

# Necessary imports
import pandas as pd
from pathlib import Path
import numpy as np
from random import shuffle
from sentencepiece import SentencePieceProcessor
import torch

# Get absolute path to raw OpenBookQA data for use as a default argument
root_path = Path(__file__).resolve().parents[1]
obqa_dir = root_path / 'data/openbookqa'

# Define preprocessing function to turn a dataset's raw JSON into a DataFrame with all necessary information to create randomized distractor prompts
def preprocess_raw(dataset: 'str', tokenizer: SentencePieceProcessor, obqa_dir: Path=obqa_dir):
    # Create paths to raw facts .txt and dataset .jsonl
    facts_path = obqa_dir / 'raw/openbook.txt'
    data_path = obqa_dir / f'raw/{dataset}_complete.jsonl'
    
    # Load data at the paths (may raise FileNotFoundError for bad inputs)
    data = pd.read_json(data_path, lines=True)
    facts = pd.read_csv(facts_path, header=None)
    facts = facts[0] # access the Series storing the facts from the dataframe above. because header=None, this series's column is simply named 0

    # Make a function that turns a row of data into a natural-language prompt. Then apply it to all rows of `data` to create prompts for each example.
    def make_prompt(example):
        stem = example['question']['stem']
        choices = ", ".join([f"Choice {choice['label']}: {choice['text']}" for choice in example['question']['choices']])
        prompt = f'Question: {stem}: {choices}. Answer: '
        return prompt
    prompts = [make_prompt(data.iloc[i]) for i in range(len(data))]
    answers = data['answerKey']

    # Find the lengths of all prompts after they are tokenized
    prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]

    # Find the row ID of the gold fact used to make the question stem of each prompt
    def find_fact_idx(prompt_idx, gold_fact):
        for i, fact in enumerate(facts):
            if gold_fact == fact:
                return i
        raise ValueError(f"No fact matches the gold fact for the prompt at index {prompt_idx}!")
    fact_indexes = [find_fact_idx(i, gold) for i, gold in enumerate(data['fact1'])]

    # When prepending facts to the prompt, we want to indicate the beginning of a new fact with the text "Fact: "
    # Apply this transformation, then find the lengths of each fact upon tokenization
    facts = [f'Fact: {fact}.' for fact in facts]
    fact_lengths = [len(tokenizer.encode(fact)) for fact in facts]

    # Create paths to saved results
    work_dir = obqa_dir / 'working'
    work_dir.mkdir(parents=False, exist_ok=True)
    prompt_save_path = work_dir / f'{dataset}.csv'
    fact_save_path = work_dir / 'facts.csv'

    # Create dataframes storing information related to each prompt/fact, then save them
    prompt_df = pd.DataFrame({
        "prompt": prompts,
        "length": prompt_lengths,
        "fact_idx": fact_indexes,
        "answer": answers
    })
    fact_df = pd.DataFrame({
        "fact": facts,
        "length": fact_lengths
    })
    prompt_df.to_csv(prompt_save_path, index=False, sep=';')
    fact_df.to_csv(fact_save_path, index=False, sep=';')


# Define function to take preprocessed data files and use them to generate full-context-length questions. Each question will include the original prompt and the gold fact used to create it.
# Random distractor facts will be sampled until no more facts can fit in the context length. Then the facts will be shuffled and prepended to the prompt to create the question. Finally, the
# question will be tokenized, and potentially left-padded to the maximum context length so that the questions can be saved to disk as a Tensor of token IDs ready to be loaded. 
def generate_questions(dataset: 'str', tokenizer: SentencePieceProcessor, obqa_dir: Path=obqa_dir, context_length: int=512, add_distractor=True, save_supporting=True):
    # Create paths to preprocessed facts and prompts dataframes
    facts_path = obqa_dir / 'working/facts.csv'
    prompts_path = obqa_dir / f'working/{dataset}.csv'
    
    # Load data at the paths (may raise FileNotFoundError for bad inputs)
    prompts = pd.read_csv(prompts_path, sep=';')
    facts = pd.read_csv(facts_path, sep=';')

    # Generate the questions and save supporting data
    questions = []         # stores the tokenized version of each question
    question_lengths = []  # stores the length of the question before left-padding to the maximum context length
    facts_used = []        # stores the indexes of the distractor (and gold) facts used in the question in the order in which they appear
    answers = []           # stores the tokenized answer choice for each question

    for _, row in prompts.iterrows():
        # Load data stored in the row
        prompt, prompt_length, gold_fact_idx, answer = row['prompt'], row['length'], row['fact_idx'], row['answer']

        # Account for size of the prompt and its gold fact
        size = prompt_length + facts.loc[gold_fact_idx, 'length']
        inc_facts = [gold_fact_idx]

        # Add facts without replacement until we exceed the maximum context length
        if add_distractor:
            while size <= context_length:
                next_fact_idx = np.random.choice(len(facts))
                if next_fact_idx in inc_facts:
                    continue
                size += facts.loc[next_fact_idx, 'length']
                inc_facts.append(next_fact_idx)
            
            # The last fact we add will always exceed the context length, so remove it
            last_fact_idx = inc_facts.pop()      # this also edits the inc_facts list
            size -= facts.loc[last_fact_idx, 'length']

            # Shuffle the facts list so that the gold fact is not always first
            shuffle(inc_facts)

        # Prepend the facts to the prompt
        question = " ".join([
            " ".join([facts.loc[idx, 'fact'] for idx in inc_facts]),
            prompt
        ])

        # Tokenize the question and store its supporting data
        question = tokenizer.encode(question)
        assert len(question) == size  # sanity check
        question_lengths.append(size)
        facts_used.append(inc_facts)
        # answers.append(tokenizer.encode(answer))
        if answer == "A":
            answers.append(0)
        elif answer == "B":
            answers.append(1)
        elif answer == "C":
            answers.append(2)
        elif answer == "D":
            answers.append(3)
        else:
            raise ValueError("Parsed an unexpected answer choice!")

        # Prepend padding tokens until the question is as large as the context length, then save the question
        padding = [tokenizer.pad_id()] * (context_length - size)
        question = padding + question
        questions.append(question)
    
    # Transform collected data to their final representations
    dtype = torch.int16 if len(tokenizer) < 2**15 - 1 else torch.int32
    questions = torch.tensor(questions, dtype=dtype)
    answers = torch.tensor(answers, dtype=dtype)
    support_df = pd.DataFrame({
        "length": question_lengths,
        "facts_used": facts_used
    })

    # Save questions and their supporting data
    difficulty = "easy"
    if add_distractor:
        difficulty = "hard"
    save_dir = obqa_dir / 'processed'
    save_dir.mkdir(parents=False, exist_ok=True)
    torch.save(questions, save_dir / f'{dataset}-{difficulty}-questions.pt')
    torch.save(answers, save_dir / f'{dataset}-{difficulty}-answers.pt')
    if save_supporting:
        support_df.to_csv(save_dir / f'{dataset}-{difficulty}-support.csv', index=False, sep=';')


if __name__ == "__main__":
    # Get path to tokenizer
    tokenizer_path = root_path / 'unigram-tokenizer/tokenizer.model'
    tokenizer_path = tokenizer_path.as_posix()
    tokenizer = SentencePieceProcessor(model_file=tokenizer_path)

    # Preprocess all the data
    preprocess_raw('train', tokenizer)
    preprocess_raw('dev', tokenizer)
    preprocess_raw('test', tokenizer)

    # Generate questions without distractors
    generate_questions('train', tokenizer, add_distractor=False, save_supporting=True)
    generate_questions('dev', tokenizer, add_distractor=False, save_supporting=True)
    generate_questions('test', tokenizer, add_distractor=False, save_supporting=True)

    # Generate questions with distractors
    generate_questions('train', tokenizer, add_distractor=True, save_supporting=True)
    generate_questions('dev', tokenizer, add_distractor=True, save_supporting=True)
    generate_questions('test', tokenizer, add_distractor=True, save_supporting=True)