import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import export_text

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

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, df['v1'], test_size=0.2, random_state=42)

# Step 6: Implement the ID3 Algorithm
# Step 7: Train the Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = clf.predict(X_test)

# Corrected part: Use vectorizer.get_feature_names_out() for feature names
plt.figure(figsize=(18, 12))
plot_tree(clf, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=['non-spam', 'spam'], rounded=True)
#plt.show()

# Export the decision tree to a Graphviz file
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=vectorizer.get_feature_names_out(),
                           class_names=['non-spam', 'spam'],
                           filled=True, rounded=True, special_characters=True)

# Visualize the Graphviz file using the graphviz library
graph = graphviz.Source(dot_data)
graph.render("spam_decision_tree", format="png")
graph.view("spam_decision_tree")

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


