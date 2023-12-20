from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load 20 Newsgroups dataset with specific categories
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(twenty_train.data)
y_train = twenty_train.target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_train_tfidf, y_train, test_size=0.3, random_state=42)

# Train the SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(x_train, y_train)

# Predict on the test set
predictions = svm_classifier.predict(x_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, predictions)
classification = classification_report(y_test, predictions, target_names=twenty_train.target_names)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification)

# New data for prediction
new_data = ["I have a question about medicine"]
x_new_tfidf = vectorizer.transform(new_data)

# Predict categories for new data
new_predictions = svm_classifier.predict(x_new_tfidf)

predicted_category = twenty_train.target_names[new_predictions[0]]
print("Predicted Category:", predicted_category)
