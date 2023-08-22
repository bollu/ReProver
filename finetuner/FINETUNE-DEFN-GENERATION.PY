#!/usr/bin/env python3.9
from torch import nn
from transformers import Seq2SeqTrainer,AutoTokenizer, T5ForConditionalGeneration,EarlyStoppingCallback, Seq2SeqTrainingArguments, AdamW, ProgressCallback
from tqdm import tqdm
import torch
import json
import pandas as pd
from datasets import  Dataset
import datasets
from nltk.translate.bleu_score import sentence_bleu
import time
from datetime import timedelta
import wandb

# TODO: 
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="lean3-codet5-fstar-finetune-definition-prediction",
    # Track hyperparameters and run metadata
    config={})

save_directory='./model-finetuned-definition-prediction' #for saving the in the current working directory

# Opening JSON file
train = open('../../llm_benchmarking/fstar_insights/lemmas_with_premises/saikat_dataset_filtered_premises/initial_data/train.json')
valid = open('../../llm_benchmarking/fstar_insights/lemmas_with_premises/saikat_dataset_filtered_premises/initial_data/validate.json')
test = open('../../llm_benchmarking/fstar_insights/lemmas_with_premises/saikat_dataset_filtered_premises/initial_data/test.json')

train_data = json.load(train)
valid_data=json.load(valid)
test_data=json.load(test)

train_df = pd.DataFrame.from_dict(train_data, orient='columns')
valid_df = pd.DataFrame.from_dict(valid_data, orient='columns')
test_df = pd.DataFrame.from_dict(test_data, orient='columns')


train_df['input'] = train_df['name'] + '<SEP>' + train_df['type']
valid_df['input'] = valid_df['name'] + '<SEP>' + valid_df['type']
test_df['input'] = test_df['name'] + '<SEP>' + test_df['type']

#changing type of data to hugging face dataset for .map() and for Trainer class

train = train_df[['input', 'definition']]
valid = valid_df[['input', 'definition']]
test = test_df[['input', 'definition']]
train_dataset = Dataset.from_pandas(train)
valid_dataset = Dataset.from_pandas(valid)
test_dataset = Dataset.from_pandas(test)


model_name="./output_dir/checkpoint-500"
tokenizer_name = "kaiyuy/leandojo-lean4-sst-byt5-small"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
max_model_length = tokenizer.model_max_length #for CodeT5 it is 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#updating the tokenizer's vocabulary file with End of Statement <EOS> Special Token:
end_of_proof_token='<EOS>' # TODO: change to tokenizer eof tken
special_tokens_dict = {'eos_token': end_of_proof_token}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


def get_label_ids(target):
    max_length=tokenizer.model_max_length
    # to train model on End of statement token. Even When model truncates longer code, EOS remain to show model the end of the statement
    # Tokenize the target text without padding to get the tokens
    encoded_tokens = tokenizer.tokenize(target)
    # Check if the total number of tokens is greater than max_length
    if len(encoded_tokens) > max_length:
        # If yes, truncate the tokens while preserving the "<EOS>" at the end
        truncated_tokens = encoded_tokens[:max_length - 1] + [encoded_tokens[-1]]
        # Convert the truncated tokens back to input_ids
        input_ids = tokenizer.convert_tokens_to_ids(truncated_tokens)
    else:
        # If no truncation needed, keep the original tokens with padding
        input_ids = tokenizer(target, max_length=max_length, padding="max_length", truncation=True).input_ids
    return input_ids

def preprocess_data(data):  
  input = data['input']
  targets=data['definition']
  target= [inp + end_of_proof_token for inp in targets]
  # inputs = [inp for inp in input + end_of_proof_token]
  model_inputs = tokenizer(input, max_length=max_model_length, padding="max_length", truncation=True)
  # encode the summaries
  labels= [get_label_ids(tar) for tar in target]
  # important: we need to replace the index of the padding tokens by -100
  # such that they are not taken into account by the CrossEntropyLoss
  labels_with_ignore_index = []
  for labels_example in labels:
    labels_example = [label if label != 0 else -100 for label in labels_example]
    labels_with_ignore_index.append(labels_example)
  
  model_inputs["labels"] = labels_with_ignore_index

  return model_inputs


#maping the dataset into batches
train_dataset = train_dataset.map(preprocess_data, batched=True)
valid_dataset = valid_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

#Trainer Class Arguments:
args = Seq2SeqTrainingArguments(
    '.',
    evaluation_strategy = "epoch",
    learning_rate=5e-5,
    do_eval=True,
    do_train=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=2,
    num_train_epochs=15,
    warmup_steps=500,
    weight_decay=0.01,
    eval_steps=100,             
    save_steps=500,  
    load_best_model_at_end=True,
)


trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

start_time = time.time()


trainer.train()

elapsed_time_secs = time.time() - start_time

print("Total Training Time: ",round(elapsed_time_secs),'seconds')

model.save_pretrained(save_directory)
trainer.save_model(save_directory)
trainer.save_state()

model = T5ForConditionalGeneration.from_pretrained(save_directory).to(device) #loading the fine-tuned model

#Testing the Model:

def get_test_results(test_dataset):
        num_beams = 15 #can chnage upto 50
        temperature = 1.0
        predictions = []
        actuals = []
        x=0
        with torch.no_grad():
            for data in test_dataset:
                bleu_scores = []
                test=data['input']
                label_=data['definition']
                input_ids = tokenizer(test, return_tensors='pt').input_ids.to(device)
                output_sequences = model.generate(input_ids, 
                                                  num_beams=num_beams,
                                                  temperature=temperature,
                                                  num_return_sequences=15,
                                                  max_length=max_model_length)

                candidate_definitions = [tokenizer.decode(seq, skip_special_tokens=True,clean_up_tokenization_spaces=True)for seq in output_sequences ]
                for candidate_sentence in candidate_definitions: #calculating sentence BLEU between label and each of the generated candiadte
                            bleu_score = sentence_bleu([candidate_sentence.split()],label_.split())
                            bleu_scores.append(bleu_score) 
                            
                # Find the index of the candidate with the highest BLEU score
                print(bleu_scores)
                highest_bleu_index = bleu_scores.index(max(bleu_scores))
                print(highest_bleu_index)
                # Get the candidate sentence with the highest BLEU score 
                best_pred = candidate_definitions[highest_bleu_index]
                predictions.append(best_pred)
                actuals.append(label_)
                print(len(predictions))
                print(len(actuals))
                x=x+1

        return predictions,actuals

#testing the trained model on test dataset
start_time = time.time()
predictions,actuals=get_test_results(test_dataset) #get model's generated definitions and actual lemma definitions
elapsed_time_secs = time.time() - start_time
print("Total Inference Time: ",round(elapsed_time_secs),'seconds')


#Calculate Sentence BLEU Score and exact Match

def calculate_sentence_bleu(candidate, reference):
    bleu_score = sentence_bleu([candidate.split()], reference.split())
    return bleu_score

def calculate_average_bleu(predictions, actuals):
    assert len(predictions) == len(actuals), "Both predictions and actuals must have the same number of definitions."

    total_sentence_bleu=[]
    num_sentences = len(predictions)

    for i in range(num_sentences):
        sentence_bleu = calculate_sentence_bleu(predictions[i],actuals[i])        
        total_sentence_bleu.append(sentence_bleu)

    
    average_sentence_bleu = sum(total_sentence_bleu) / num_sentences
    return average_sentence_bleu,total_sentence_bleu


average_sentence_bleu,total_sentence_bleu= calculate_average_bleu(predictions, actuals)

print("Average Sentence BLEU:", round(average_sentence_bleu*100), '%')
print("Sentence BLEU for each lemma:    ",total_sentence_bleu)


predictions_tokens = [prediction.split() for prediction in predictions]
labels_tokens = [label.split() for label in actuals]

exact_match_count = 0
for prediction_tokens, label_tokens in zip(predictions_tokens, labels_tokens):
    if prediction_tokens == label_tokens:
        exact_match_count += 1
# Calculate the exact match score
exact_match_score = exact_match_count / len(predictions_tokens)

print(f"Exact Match Score: {exact_match_score:.2f}")

print('Generated lemmas: ',predictions)
print('Actual lemmas: ',actuals)
trainer.state.log_history
