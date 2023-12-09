import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz
from imblearn.under_sampling import RandomUnderSampler

# Step 1: Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Step 2: Check the Loaded dataset
print(df.head())
print("Columns:", df.columns)
print("Missing values:\n", df.isnull().sum())
print("Class distribution:\n", df['v1'].value_counts())

# Step 3: Preprocess the Data
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Step 4: Feature Extraction
# Using CountVectorizer to convert text data to a format suitable for machine learning
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])

# Step 5: Undersample the Majority Class
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, df['v1'])

# Step 6.1: Split into Training (70%) and Temporary Data (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Step 6.2: Split Temporary Data into Validation (50%) and Test (50%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 7: Implement the ID3 Algorithm
# Step 8: Train the Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Step 9: Evaluate the Model on Validation Set
y_val_pred = clf.predict(X_val)

# Corrected part: Use vectorizer.get_feature_names_out() for feature names
plt.figure(figsize=(18, 12))
plot_tree(clf, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=['non-spam', 'spam'], rounded=True)
# plt.show()

# Metrics for Validation Set
print("Accuracy on Validation Set:", accuracy_score(y_val, y_val_pred))
print("Classification Report on Validation Set:\n", classification_report(y_val, y_val_pred))

# Step 10: Evaluate the Model on Test Set
y_test_pred = clf.predict(X_test)

# Metrics for Test Set
print("Accuracy on Test Set:", accuracy_score(y_test, y_test_pred))
print("Classification Report on Test Set:\n", classification_report(y_test, y_test_pred))
