import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def balance_data(df, target_column):
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
    return pd.concat([df_majority, df_minority_upsampled])

def preprocess_data(filepath, target_column):
    df = load_data(filepath)
    df['cleaned_text'] = df['text'].apply(clean_text)
    df_balanced = balance_data(df, target_column)
    return train_test_split(df_balanced['cleaned_text'], df_balanced[target_column], test_size=0.2, random_state=42)
