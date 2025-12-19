import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("data\spam_Emails_data.csv")
copy = data.copy()
print(copy.head())
print("data loaded✅")

copy = copy.dropna(subset=['text'])
print(f"Dataset shape after dropping missing rows: {copy.shape}")
X = copy['text']   
y = copy['label'] 

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
name = "LogisticRegression"
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print(f"\n===== {name} =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

new_emails = [
    "Congratulations! You've won a free gift card. Click here to claim.",#spam
    "Hi team, the meeting is scheduled tomorrow at 10am. Please confirm.",#notspam
    "Urgent: Your account will be suspended if you do not verify now!"#spam
]

new_vec = vectorizer.transform(new_emails)

predictions = model.predict(new_vec)

label_map = {1: "spam", 0: "non-spam"}

for email, pred in zip(new_emails, predictions):
    print(f"Email: {email[:50]}... => Prediction: {label_map[pred]}")

