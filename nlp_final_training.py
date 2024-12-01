import pandas as pd
import torch
from datasets import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from evaluate import load

train_data = pd.read_parquet('train_data.parquet')
val_data = pd.read_parquet('val_data.parquet')
test_data = pd.read_parquet('test_data.parquet')

print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.get_device_name(0))  # Name of the first GPU

# Convert Pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

# Load BART model and tokenizer
model_name = "facebook/bart-large-xsum"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def preprocess_data(batch):
    inputs = tokenizer(batch['document'], max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
    outputs = tokenizer(batch['summary'], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    inputs['labels'] = outputs['input_ids']
    return inputs

# Preprocess training data using Hugging Face Dataset map function
tokenized_train = train_dataset.map(preprocess_data, batched=True)
tokenized_val = val_dataset.map(preprocess_data, batched=True)
tokenized_test = test_dataset.map(preprocess_data, batched=True)

model = model.to('cuda')

training_args = TrainingArguments(
    output_dir="./results", # Specifies the directory where model checkpoints, logs, and outputs will be saved during training. Useful for resuming training later or for deployment.
    evaluation_strategy="epoch", #Indicates when evaluation should be performed. epoch: Evaluates at the end of every training epoch. steps: Evaluates every eval_steps (e.g., every 500 steps). no: Skips evaluation
    learning_rate=5e-5, # Sets the learning rate for the optimizer. 5e-5 is a common default for fine-tuning transformer models.
    per_device_train_batch_size=16, # The batch size for training on each device (e.g., per GPU or TPU core). If using 2 GPUs, the effective batch size becomes 2 x num_gpus.
    per_device_eval_batch_size=16, # The batch size for evaluation, handled similarly to training batch size.
    num_train_epochs=5, # The number of complete passes (epochs) through the entire training dataset.
    save_steps=10000, # Saves a checkpoint of the model every 10,000 steps. This is useful for resuming training after interruptions.
    save_total_limit=2, # Limits the number of saved checkpoints. The oldest checkpoints are deleted once the limit is reached.
    fp16=True,  # Enable mixed precision for faster training
    remove_unused_columns=True,
    gradient_accumulation_steps=2,  # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
    dataloader_num_workers=32,  # Adjust based on available CPU cores
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)
trainer.train()

# Save model and tokenizer
model.save_pretrained("./trained_model_bart_large_on_full_dataset")
tokenizer.save_pretrained("./trained_tokenizer_bart_large_on_full_dataset")

metric = load("rouge")

def evaluate_summaries(model, tokenizer, data):
    summaries = []
    for sample in data:
        inputs = tokenizer(sample['document'], return_tensors="pt", truncation=True, max_length=1024)

        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output = model.generate(inputs['input_ids'], max_length=128, num_beams=4, early_stopping=True)
        summaries.append(tokenizer.decode(output[0], skip_special_tokens=True))
    return summaries

# Get predictions
test_summaries = evaluate_summaries(model, tokenizer, test_dataset)

# Compute ROUGE
results = metric.compute(predictions=test_summaries, references=test_dataset['summary'])
print("ROUGE Scores:", results)

