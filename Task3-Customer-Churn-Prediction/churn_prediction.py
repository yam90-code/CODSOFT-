# Task 3 - Customer Churn Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("   CUSTOMER CHURN PREDICTION")
print("   CodSoft ML Internship - Task 3")
print("=" * 50)


print("\n[1] Loading Dataset...")
df = pd.read_csv('Churn_Modelling.csv')
print(f"✓ Total customers: {len(df)}")
print(f"✓ Columns: {list(df.columns)}")


print("\n[2] Exploring Data...")
print(f"\nChurn Distribution:")
print(df['Exited'].value_counts())
print(f"\nChurn percentage: {df['Exited'].mean()*100:.2f}%")


print("\n[3] Cleaning Data...")
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df = df.dropna()
print(f"✓ Data cleaned! Shape: {df.shape}")


print("\n[4] Encoding Data...")
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Geography'] = le.fit_transform(df['Geography'])
print("✓ Encoding done!")


print("\n[5] Preparing Data...")
X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print(f"✓ Training: {len(X_train)} | Test: {len(X_test)}")


print("\n[6] Scaling Data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Done!")


print("\n[7] Training Models...")


print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"  ✓ Logistic Regression: {lr_acc*100:.2f}%")


print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"  ✓ Random Forest: {rf_acc*100:.2f}%")


print("  Training Gradient Boosting...")
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"  ✓ Gradient Boosting: {gb_acc*100:.2f}%")


print("\n[8] Best Model Results...")
accuracies = {
    'Logistic Regression': lr_acc,
    'Random Forest': rf_acc,
    'Gradient Boosting': gb_acc
}
best_name = max(accuracies, key=accuracies.get)
print(f"✓ Best Model: {best_name}")
print(f"✓ Accuracy: {accuracies[best_name]*100:.2f}%")

if best_name == 'Random Forest':
    best_pred = rf_pred
elif best_name == 'Gradient Boosting':
    best_pred = gb_pred
else:
    best_pred = lr_pred

print("\nClassification Report:")
print(classification_report(y_test, best_pred))


print("\n[9] Creating Graphs...")


plt.figure(figsize=(8, 5))
df['Exited'].value_counts().plot(
    kind='bar', color=['steelblue', 'red']
)
plt.title('Customer Churn Distribution')
plt.xlabel('0=No Churn  1=Churn')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('churn_distribution.png')
print("✓ Saved: churn_distribution.png")


plt.figure(figsize=(8, 5))
names = list(accuracies.keys())
accs = [v*100 for v in accuracies.values()]
bars = plt.bar(names, accs,
               color=['steelblue', 'coral', 'green'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for bar, acc in zip(bars, accs):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f'{acc:.2f}%',
        ha='center'
    )
plt.tight_layout()
plt.savefig('model_comparison.png')
print("✓ Saved: model_comparison.png")


plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d',
            cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("✓ Saved: confusion_matrix.png")


plt.figure(figsize=(12, 6))
feature_imp = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
feature_imp.plot(kind='bar', color='steelblue')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✓ Saved: feature_importance.png")

plt.show()


results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': best_pred
})
results.to_csv('churn_predictions.csv', index=False)
print("\n✓ Saved: churn_predictions.csv")

print("\n" + "=" * 50)
print("   TASK 3 COMPLETED SUCCESSFULLY! 🎉")
print("=" * 50)