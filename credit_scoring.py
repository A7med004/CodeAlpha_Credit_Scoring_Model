import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Simulate dataset
def simulate_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    income = np.random.normal(50000, 15000, n_samples)
    debts = np.random.normal(15000, 5000, n_samples)
    payment_history = np.random.randint(0, 2, n_samples)  # 0: bad, 1: good
    age = np.random.randint(21, 65, n_samples)
    # Creditworthy if income high, debts low, good payment history
    creditworthy = (
        (income > 45000).astype(int) +
        (debts < 20000).astype(int) +
        payment_history
    )
    creditworthy = (creditworthy >= 2).astype(int)
    data = pd.DataFrame({
        'income': income,
        'debts': debts,
        'payment_history': payment_history,
        'age': age,
        'creditworthy': creditworthy
    })
    return data

data = simulate_data()
print('First 5 rows of simulated data:')
print(data.head())

# 2. Feature engineering (debt-to-income ratio)
data['debt_to_income'] = data['debts'] / data['income']

# 3. Split data
X = data.drop('creditworthy', axis=1)
y = data['creditworthy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure X_train and X_test are DataFrames, y_train and y_test are Series
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train, name='creditworthy')
y_test = pd.Series(y_test, name='creditworthy')

# Save train and test datasets as CSV
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv('train_data.csv', index=False)

test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('test_data.csv', index=False)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train models
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    results = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    return results, classification_report(y_test, y_pred)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

all_results = {}
for name, model in models.items():
    results, report = train_and_evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
    all_results[name] = results
    print(f'\n{name} Classification Report:\n{report}')

# 6. Display results
df_results = pd.DataFrame(all_results).T
print('\nModel Comparison:')
print(df_results)

# 7. Plot results
plt.figure(figsize=(8, 5))
sns.barplot(data=df_results.reset_index().melt(id_vars='index'), x='index', y='value', hue='variable')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.legend(title='Metric')
plt.tight_layout()
plt.show() 