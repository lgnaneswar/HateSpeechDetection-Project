# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
import pytesseract
from PIL import Image
from sklearn.metrics import accuracy_score 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize NLTK lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stopword = set(stopwords.words('english'))

# Function to clean text data with lemmatization
def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(rf'[{string.punctuation}]', '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with numbers
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stopword]
    return " ".join(text)

# Set up pytesseract path
pytesseract.pytesseract.tesseract_cmd = r"tesseract.exe"

# Function to extract text from image using pytesseract
def extract_text(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

# Load dataset
data = pd.read_csv("twitter.csv")
# data = data.head(6000)
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
data = data[["tweet", "labels"]]
data["tweet"] = data["tweet"].apply(clean)

# Split data into features (X) and labels (y)
x = np.array(data["tweet"])
y = np.array(data["labels"])

# Feature extraction
cv = CountVectorizer()
X = cv.fit_transform(x)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Model training
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Compute Accuracy_score
ac=accuracy_score(y_test,y_pred)

# Display Accuracy Score
print("Accuracy Score\n",ac)