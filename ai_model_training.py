import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load datasets
df1 = pd.read_csv('data sets/spam_ham_dataset.csv')
df2 = pd.read_csv('data sets/Spam Email raw text for NLP.csv')

# Preprocess the datasets as done before
# Combine both datasets
df1['text'] = df1['text'].astype(str)  # Ensure text is string
df2['MESSAGE'] = df2['MESSAGE'].astype(str)  # Ensure text is string

# Combine both datasets for model training
df = pd.concat([df1[['text', 'label_num']], df2[['MESSAGE', 'CATEGORY']].rename(columns={'CATEGORY': 'label_num', 'MESSAGE': 'text'})], axis=0)

# Split into training and testing datasets
X = df['text']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizer (TF-IDF + n-grams)
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')

# Random Forest Model
model = RandomForestClassifier()

# Hyperparameters for tuning (for GridSearchCV)
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, None]
}

# Create pipeline with vectorizer and classifier
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', model)
])

# Perform Grid Search CV
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Make predictions and evaluate the model
y_pred = best_model.predict(X_test)

# Print accuracy
from sklearn.metrics import accuracy_score
print(f"Accuracy on Test Set: {accuracy_score(y_test, y_pred)}")

# Save the trained model (Random Forest + Vectorizer) into a pickle file
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)
    print("Random Forest model saved as 'random_forest_model.pkl'")

# Save the vectorizer into a pickle file
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
    print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")
