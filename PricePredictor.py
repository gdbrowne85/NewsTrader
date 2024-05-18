from transformers import BertTokenizer, BertModel
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
from collections import Counter
from itertools import chain

# Assuming the DataFrame is loaded from 'updated_pre_processed_news.xlsx'
df = pd.read_excel('updated_pre_processed_news.xlsx')
print("excel file imported; df created")

# Keeping the specified columns and dropping rows with missing values
df = df[['title', 'text', 'open', 'sentiment_score']].dropna()
print('Inapplicable columns dropped and rows with missing values dropped from dataframe')
negative_df = df[(df['sentiment_score'] < -5)]
positive_df = df[(df['sentiment_score'] > 5)]
print(negative_df)
print(positive_df)

# Step 1: Concatenate 'text' and 'title' fields
negative_df['combined'] = negative_df['title'] + ' ' + negative_df['text']

# Step 2: Tokenize the concatenated strings
# For simplicity, this example just splits by spaces and removes punctuation.
# For more accurate tokenization, consider using an NLP library like nltk or spaCy.
negative_df['tokens'] = negative_df['combined'].str.replace(r'[^\w\s]', '', regex=True).str.lower().str.split()

# Step 3: Count occurrences of each unique word
# Flatten the list of tokens into a single list and count occurrences
word_counts = Counter(chain.from_iterable(negative_df['tokens']))

print(word_counts)
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print('Tokenizer loaded')

# Load pre-trained model (weights) and put model in evaluation mode
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
print('Pretrained model loaded and put into evaluation mode')

np.set_printoptions(threshold=np.inf)

# Function to generate embeddings for a given text
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.cpu().numpy()

# Generating embeddings for each text in the DataFrame
print('Generating embeddings for text data')
# Assume df['text'] is your dataset
texts = df['text']

start_time = time.time()
embeddings = []  # Initialize an empty list to store embeddings

for i, text in enumerate(texts):
    start_iteration_time = time.time()

    # Process the text to get embeddings
    embedding = get_embeddings(text)
    embeddings.append(embedding)

    # Calculate the time taken for this iteration
    iteration_time = time.time() - start_iteration_time

    # Update the user on the progress
    percent_complete = (i + 1) / len(texts) * 100
    total_time = time.time() - start_time
    avg_time_per_item = total_time / (i + 1)
    estimated_time_remaining = avg_time_per_item * (len(texts) - (i + 1))

    print(f"Processed {i + 1}/{len(texts)} texts ({percent_complete:.2f}% complete). "
          f"Estimated time remaining: {estimated_time_remaining:.2f} seconds.")

# Convert the list of embeddings to a numpy array
embeddings = np.vstack(embeddings)
print(embeddings)
# Splitting the data into training and test sets, but now y is 'sentiment_score' directly without categorization
X_train, X_test, y_train, y_test = train_test_split(embeddings, df['sentiment_score'], test_size=0.2)

# Transforming data to polynomial features
degree = 2  # Example degree of polynomial features
poly_features = PolynomialFeatures(degree=degree)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

# Training a Linear Regression model on the polynomial features
model = LinearRegression()
model.fit(X_poly_train, y_train)
print('Regression model trained on polynomial features')

# Evaluating the model
from sklearn.metrics import mean_squared_error

y_pred = model.predict(X_poly_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE) on test set: {mse:.2f}')
results = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred})
print(results)
