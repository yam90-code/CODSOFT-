#movie genre classification-Task 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("   MOVIE GENRE CLASSIFICATION")
print("   CodSoft ML Internship - Task 1")
print("=" * 50)

print("\n[1] Loading Dataset...")
train_data = pd.read_csv(
    'train_data.txt',
    sep=':::',
    engine='python',
    header=None,
    names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
)

test_data = pd.read_csv(
    'test_data.txt',
    sep=':::',
    engine='python',
    header=None,
    names=['ID', 'TITLE', 'DESCRIPTION']
)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Unique Genres: {train_data['GENRE'].nunique()}")


print("\n[2] Cleaning Data...")
train_data = train_data.dropna()
train_data['DESCRIPTION'] = train_data['DESCRIPTION'].str.lower()
train_data['DESCRIPTION'] = train_data['DESCRIPTION'].str.replace(
    '[^a-zA-Z\s]', '', regex=True
)
print("Data cleaned!")


print("\n[3] Preparing Data...")
X = train_data['DESCRIPTION']
y = train_data['GENRE'].str.strip()

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print(f"Train: {len(X_train)} | Validation: {len(X_val)}")


print("\n[4] Converting Text to Numbers...")
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
print("Text converted!")


print("\n[5] Training Models...")


nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_val_tfidf)
nb_acc = accuracy_score(y_val, nb_pred)
print(f"Naive Bayes Accuracy: {nb_acc*100:.2f}%")


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_val_tfidf)
lr_acc = accuracy_score(y_val, lr_pred)
print(f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")

print("\n[6] Best Model Results...")
if lr_acc > nb_acc:
    best_pred = lr_pred
    best_model = lr_model
    print("Best Model: Logistic Regression")
else:
    best_pred = nb_pred
    best_model = nb_model
    print("Best Model: Naive Bayes")

print("\nClassification Report:")
print(classification_report(y_val, best_pred))


print("\n[7] Creating Graphs...")

plt.figure(figsize=(12, 6))
train_data['GENRE'].str.strip().value_counts().head(10).plot(
    kind='bar', color='steelblue'
)
plt.title('Top 10 Movie Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('genre_distribution.png')
print("Saved: genre_distribution.png")


plt.figure(figsize=(8, 5))
model_names = ['Naive Bayes', 'Logistic Regression']
accuracies = [nb_acc*100, lr_acc*100]
bars = plt.bar(model_names, accuracies, color=['steelblue', 'coral'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f'{acc:.2f}%',
        ha='center'
    )
plt.tight_layout()
plt.savefig('model_comparison.png')
print("Saved: model_comparison.png")

plt.show()


print("\n[8] Sample Predictions...")
X_test_tfidf = tfidf.transform(test_data['DESCRIPTION'])
predictions = best_model.predict(X_test_tfidf)

print("-" * 40)
for i in range(5):
    print(f"Movie : {test_data['TITLE'][i].strip()}")
    print(f"Genre : {predictions[i].strip()}")
    print("-" * 40)


results_df = pd.DataFrame({
    'Title': test_data['TITLE'],
    'Predicted_Genre': predictions
})
results_df.to_csv('predictions.csv', index=False)
print("\nPredictions saved to predictions.csv")

print("\n" + "=" * 50)
print("   TASK 1 COMPLETED SUCCESSFULLY!")
print("=" * 50)