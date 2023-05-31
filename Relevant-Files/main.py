# Importing required libraries
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Folder Paths
left_folder = 'LeftFolder'
right_folder = 'RightFolder'

# Get file paths from the left and right folders
left_files = os.listdir(left_folder)
right_files = os.listdir(right_folder)

# Load stop words and initialize lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Prepare left document
left_file_path = os.path.join(left_folder, left_files[0])
with open(left_file_path, 'r') as left_file:
    left_document = left_file.read()

# Preprocess left document
left_tokens = word_tokenize(left_document.lower())
left_tokens = [lemmatizer.lemmatize(token) for token in left_tokens if token.isalnum()]
left_tokens = [token for token in left_tokens if token not in stop_words]
left_preprocessed = ' '.join(left_tokens)

# Prepare right documents
right_documents = []
for right_file in right_files:
    right_file_path = os.path.join(right_folder, right_file)
    with open(right_file_path, 'r') as file:
        right_document = file.read()
        right_documents.append(right_document)

# Preprocess right documents
right_tokens_list = []
for right_document in right_documents:
    right_tokens = word_tokenize(right_document.lower())
    right_tokens = [lemmatizer.lemmatize(token) for token in right_tokens if token.isalnum()]
    right_tokens = [token for token in right_tokens if token not in stop_words]
    right_tokens_list.append(' '.join(right_tokens))

# Vectorize documents
vectorizer = TfidfVectorizer()
vectorized_documents = vectorizer.fit_transform([left_preprocessed] + right_tokens_list)
left_vector = vectorized_documents[0]
right_vectors = vectorized_documents[1:]

# Calculate cosine similarity between left and right documents
similarities = cosine_similarity(left_vector, right_vectors)
similarities = similarities[0]

# Get top 3 relevant files
top_indices = similarities.argsort()[-3:][::-1]
top_files = [right_files[i] for i in top_indices]

# Print the top 3 relevant files and their rankings
for i, file in enumerate(top_files, 1):
    print(f"Rank {i}: {file}")
