import json
from transformers import BertTokenizer, BertForTokenClassification
import torch
import arabic_reshaper
from bidi.algorithm import get_display
import codecs

# Load the JSON dataset
# with open('CANERCorpus.json', 'r', encoding='utf-8') as file:
dataset = json.load(codecs.open('CANERCorpus.json', 'r', encoding='ISO-8859-1'))

print(dataset[0])
# Sample data from the dataset (modify based on your JSON structure)
# sample_text = dataset[0]
# reshaped_text = arabic_reshaper.reshape(sample_text)    # correct its shape
# bidi_text = get_display(reshaped_text)
# print(bidi_text)

# Load pre-trained Arabic BERT model and tokenizer
model_name = 'asafaya/bert-base-arabic'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=2)  # Change num_labels based on your entity types

# Tokenize the text and get token IDs
tokens = tokenizer.tokenize(dataset[0]['Word'])
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([token_ids])

# Get model predictions
with torch.no_grad():
    outputs = model(input_ids)

# Get predicted labels (NER tags)
predicted_labels = torch.argmax(outputs.logits, dim=2)[0].tolist()
print('Predicted entities: ', predicted_labels)
predicted_entities = [tokenizer.convert_ids_to_tokens(i) for i in predicted_labels]
print('Predicted entities: ', predicted_entities)
# Map BERT's token-level labels to entities
entities = []
current_entity = ""
for token, label in zip(tokens, predicted_entities):
    if label.startswith('B'):
        if current_entity:
            entities.append(current_entity)
        current_entity = token
    elif label.startswith('I'):
        current_entity += " " + token if current_entity else token
    else:
        if current_entity:
            entities.append(current_entity)
            current_entity = ""

# Print the identified entities
print("Identified entities:")
print(entities)
