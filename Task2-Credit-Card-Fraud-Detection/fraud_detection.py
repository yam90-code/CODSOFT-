
# Task 2 - Credit Card Fraud Detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("   CREDIT CARD FRAUD DETECTION")
print("   CodSoft ML Internship - Task 2")
print("=" * 50)


print("\n[1] Loading Dataset...")
train_df = pd.read_csv('fraudTrain.csv')
test_df = pd.read_csv('fraudTest.csv')

print(f"✓ Training samples: {len(train_df)}")
print(f"✓ Test samples: {len(test_df)}")
print(f"✓ Columns: {list(train_df.columns)}")


print("\n[2] Exploring Data...")
print(f"\nFraud vs Legitimate in Training:")
print(train_df['is_fraud'].value_counts())
print(f"\nFraud percentage: {train_df['is_fraud'].mean()*100:.2f}%")

print("\n[3] Preparing Data...")


features = ['amt', 'lat', 'long', 'city_pop', 
            'unix_time', 'merch_lat', 'merch_long']

X_train = train_df[features]
y_train = train_df['is_fraud']

X_test = test_df[features]
y_test = test_df['is_fraud']


fraud = train_df[train_df['is_fraud']==1]
legit = train_df[train_df['is_fraud']==0].sample(
    n=len(fraud)*3, random_state=42
)
balanced = pd.concat([fraud, legit])

X_bal = balanced[features]
y_bal = balanced['is_fraud']

X_tr, X_val, y_tr, y_val = train_test_split(
    X_bal, y_bal,
    test_size=0.2,
    random_state=42
)

print(f"✓ Training: {len(X_tr)} | Validation: {len(X_val)}")


print("\n[4] Scaling Data...")
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
print("✓ Done!")


print("\n[5] Training Models...")


print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_tr_scaled, y_tr)
lr_pred = lr.predict(X_val_scaled)
lr_acc = accuracy_score(y_val, lr_pred)
print(f"  ✓ Logistic Regression: {lr_acc*100:.2f}%")


print("  Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_tr_scaled, y_tr)
dt_pred = dt.predict(X_val_scaled)
dt_acc = accuracy_score(y_val, dt_pred)
print(f"  ✓ Decision Tree: {dt_acc*100:.2f}%")


print("  Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=50, random_state=42
)
rf.fit(X_tr_scaled, y_tr)
rf_pred = rf.predict(X_val_scaled)
rf_acc = accuracy_score(y_val, rf_pred)
print(f"  ✓ Random Forest: {rf_acc*100:.2f}%")


print("\n[6] Best Model Results...")
accuracies = {
    'Logistic Regression': lr_acc,
    'Decision Tree': dt_acc,
    'Random Forest': rf_acc
}
best_name = max(accuracies, key=accuracies.get)
print(f"✓ Best Model: {best_name}")
print(f"✓ Accuracy: {accuracies[best_name]*100:.2f}%")

if best_name == 'Random Forest':
    best_pred = rf_pred
elif best_name == 'Decision Tree':
    best_pred = dt_pred
else:
    best_pred = lr_pred

print("\nClassification Report:")
print(classification_report(y_val, best_pred))


print("\n[7] Creating Graphs...")

plt.figure(figsize=(8, 5))
train_df['is_fraud'].value_counts().plot(
    kind='bar', color=['steelblue', 'red']
)
plt.title('Fraud vs Legitimate Transactions')
plt.xlabel('0=Legitimate  1=Fraud')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('fraud_distribution.png')
print("✓ Saved: fraud_distribution.png")


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
cm = confusion_matrix(y_val, best_pred)
sns.heatmap(cm, annot=True, fmt='d',
            cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("✓ Saved: confusion_matrix.png")

plt.show()


results = pd.DataFrame({
    'Actual': y_val.values,
    'Predicted': best_pred
})
results.to_csv('fraud_predictions.csv', index=False)
print("\n✓ Saved: fraud_predictions.csv")

print("\n" + "=" * 50)
print("   TASK 2 COMPLETED SUCCESSFULLY BY YAM! 🎉")
print("=" * 50)