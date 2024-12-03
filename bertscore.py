import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bert_score import score

test_data = pd.read_parquet('test_data.parquet')


test_data = test_data.iloc[:2000]

print(test_data.info())


# Prepare inputs for the model
test_documents = test_data["document"].tolist()  # Replace with the column containing documents
reference_summaries = test_data["summary"].tolist()  # Replace with the column containing reference summaries


# Specify the path to your saved model directory
model_path = "./saved_model"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./trained_tokenizer_bart_large_on_full_dataset")

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained("./trained_model_bart_large_on_full_dataset")


generated_summaries = []

for document in test_documents:
    # Tokenize the input document
    inputs = tokenizer(document, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], max_length=128, min_length=30, num_beams=4, length_penalty=2.0)
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Append to list
    generated_summaries.append(generated_summary)
    if len(generated_summaries) % 100 == 0 : print(len(generated_summaries))


# Compute BERTScore
P, R, F1 = score(generated_summaries, reference_summaries, lang="en", verbose=True)

# Display the average BERTScore metrics
print("--------------------------BertScore for bart-large-xsum model--------------------------")
print(f"Average Precision: {P.mean().item():.4f}")
print(f"Average Recall: {R.mean().item():.4f}")
print(f"Average F1 Score: {F1.mean().item():.4f}")



# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./trained_tokenizer_t5-small")

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained("./trained_model_t5-small")

# Generate summaries
generated_summaries = []

for document in test_documents:
    # Tokenize the input document
    inputs = tokenizer(document, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], max_length=128, min_length=30, num_beams=4, length_penalty=2.0)
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Append to list
    generated_summaries.append(generated_summary)
    if len(generated_summaries) % 100 == 0 : print(len(generated_summaries))

# Compute BERTScore
P, R, F1 = score(generated_summaries, reference_summaries, lang="en", verbose=True)

# Display the average BERTScore metrics for t5 small model
print("--------------------------BertScore for t5-small model--------------------------")
print(f"Average Precision: {P.mean().item():.4f}")
print(f"Average Recall: {R.mean().item():.4f}")
print(f"Average F1 Score: {F1.mean().item():.4f}")