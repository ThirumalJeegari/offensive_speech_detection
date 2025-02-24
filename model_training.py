from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

def train_logistic_regression(X_train, y_train):
    model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
    model.fit(X_train, y_train)
    return model

def train_bert(X_train, y_train):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16)
    return model
