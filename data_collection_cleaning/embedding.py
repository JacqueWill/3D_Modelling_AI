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

# Print the embeddings
print(embeddings)
