import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample dataset (replace with your actual dataset)
data = {
    "text": [
        "I hate you!",
        "You're an amazing person!",
        "This is so stupid",
        "I love this!",
        "You are the worst!",
        "Great job on the project!",
        "Go away, loser!"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1]  # 1 = Offensive, 0 = Not Offensive
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")
