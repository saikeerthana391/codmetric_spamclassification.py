# spam_classifier.py

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK stopwords (only first run will download)
nltk.download('stopwords')

# ✅ Load dataset correctly (tab separated, no headers in your file)
df = pd.read_csv("spam.csv", sep="\t", encoding="latin-1", header=None)

# ✅ Rename columns
df.columns = ["label", "message"]

print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nLabel Distribution:")
print(df['label'].value_counts())

# ✅ Map labels: ham -> 0, spam -> 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# ✅ Text preprocessing function
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = text.split()
    # Remove stopwords + stemming
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# ✅ Apply preprocessing
df['cleaned_message'] = df['message'].apply(preprocess_text)

print("\nSample cleaned messages:")
print(df[['message', 'cleaned_message']].head())

# ✅ Split dataset
X = df['cleaned_message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ✅ Predictions
y_pred = model.predict(X_test_vec)

# ✅ Evaluation
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
