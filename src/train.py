import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import BertTokenizer, BertForTokenClassification, BasicTokenizer
from datasets import load_dataset, load_metric
import pandas as pd
import numpy as np
from transformers import DataCollatorForTokenClassification
import evaluate

seqeval = evaluate.load("seqeval")


df = pd.read_csv('train.csv')
arr = []
for s in df['tags']:
    arr += s.split(' ')

arr = np.unique(arr)
id2label = dict(zip([i for i in range(len(arr))], [elem for elem in arr]))
label2id = dict(zip([elem for elem in arr], [i for i in range(len(arr))]))
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

def map_fn(x):
    tokenized_inputs = tokenizer(x["sentence"].split(' '), is_split_into_words=True, truncation=True)
    x['tags'] = [label2id[elem] for elem in x['tags'].split(' ')]
    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label = x['tags']
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [arr[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [arr[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

dataset = load_dataset("csv", split='train',data_files="train.csv")


dataset = dataset.train_test_split(test_size=0.001, train_size=0.1)
dataset = dataset.map(map_fn)


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-multilingual-cased", num_labels=len(arr), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="my_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()




