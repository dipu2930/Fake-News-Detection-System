import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
# Load dataset
df_fake = pd.read_csv(r"C:\Users\DIPANJAN AICH\Desktop\Fake-News-Detection\dataset\Fake.csv\Fake.csv", encoding="utf-8")
df_real = pd.read_csv(r"C:\Users\DIPANJAN AICH\Desktop\Fake-News-Detection\dataset\True.csv\True.csv", encoding="utf-8")
df_fake["label"] = 1  # Fake news
df_real["label"] = 0  # Real news

# Combine both datasets
df = pd.concat([df_fake, df_real], axis=0).reset_index(drop=True)

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

# Apply preprocessing
df["text"] = df["text"].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Machine Learning Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate Model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model and vectorizer saved successfully!")
