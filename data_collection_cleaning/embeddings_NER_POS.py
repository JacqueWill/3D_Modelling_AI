import nltk
import torch
from transformers import BertTokenizer, BertModel

# Load BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text
text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')

# Generate the token embeddings
with torch.no_grad():
    model_output = model(tokens)

# Extract the last hidden state of the tokens as the embeddings
embeddings = model_output.last_hidden_state[0]

# Convert embeddings to list of token strings
token_strings = tokenizer.convert_ids_to_tokens(tokens[0])

# Perform Named Entity Recognition
ner_tags = nltk.ne_chunk(nltk.pos_tag(token_strings))

# Perform Part of Speech Tagging
pos_tags = nltk.pos_tag(token_strings)

# Print the results
print("Named Entity Recognition tags:")
print(ner_tags)
print("Part of Speech tags:")
print(pos_tags)
