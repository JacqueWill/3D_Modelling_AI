import torch
import nltk
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load training data
with open('train.txt', 'r') as f:
    train_data = f.readlines()

# Initialize lists for NER tags, POS tags, and embeddings
ner_tags = []
pos_tags = []
embeddings = []

# Process each training example
for example in train_data:
    # Extract the text from the example
    text = example.strip()

    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')

    # Generate the token embeddings
    with torch.no_grad():
        model_output = model(tokens)

    # Extract the last hidden state of the tokens as the embeddings
    example_embeddings = model_output.last_hidden_state[0].numpy()

    # Convert embeddings to list of token strings
    token_strings = tokenizer.convert_ids_to_tokens(tokens[0])

    # Perform Named Entity Recognition
    example_ner_tags = nltk.ne_chunk(nltk.pos_tag(token_strings))

    # Perform Part of Speech Tagging
    example_pos_tags = nltk.pos_tag(token_strings)

    # Append the NER tags, POS tags, and embeddings to the corresponding lists
    ner_tags.append(example_ner_tags)
    pos_tags.append(example_pos_tags)
    embeddings.append(example_embeddings)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    embeddings,  # input features
    y,           # target variable for 3D scenario prediction
    test_size=0.2,
    random_state=42
)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
score = model.score(X_test, y_test)
print("Model accuracy:", score)
