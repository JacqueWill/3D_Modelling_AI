import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Example text input
text = "The quick brown fox jumps over the lazy dog. The dog barks loudly."

# Tokenize the text into sentences and words
sentences = sent_tokenize(text)
words = word_tokenize(text)

# Remove stop words and punctuation
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]

# Stem or lemmatize the remaining words
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmed_words = [ps.stem(word) for word in filtered_words]
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

# Print the results
print("Original Text:", text)
print("Sentences:", sentences)
print("Filtered Words:", filtered_words)
print("Stemmed Words:", stemmed_words)
print("Lemmatized Words:", lemmatized_words)
