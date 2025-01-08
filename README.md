# CODESOFT_5
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample Data
plot_summaries = [
    "A group of friends embark on a thrilling adventure.",
    "A heartwarming tale of love and loss.",
    "An action-packed story of superheroes saving the world.",
    "A detective solves a mysterious murder case.",
    "A hilarious comedy about a family reunion gone wrong."
]
genres = ['Adventure', 'Romance', 'Action', 'Mystery', 'Comedy']

# Convert genres to numerical labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(genres)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(plot_summaries)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Predict genre for a new movie plot
new_plot = ["A thrilling adventure of a space crew exploring the unknown."]
new_X = vectorizer.transform(new_plot)
predicted_genre = le.inverse_transform(classifier.predict(new_X))
print("Predicted Genre:", predicted_genre[0])
