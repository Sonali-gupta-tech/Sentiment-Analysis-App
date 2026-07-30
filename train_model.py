import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("dataset/sentiment_dataset.csv")

# Features & Labels
X = df["review"]
y = df["sentiment"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Model
model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print(f"Model Accuracy : {accuracy*100:.2f}%")

# Save Model
with open("model/sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save Vectorizer
with open("model/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("✅ Model Saved Successfully")