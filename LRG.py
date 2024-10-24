import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('dataset_thyroid_sick.csv')
df.head()

df = df.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'TBG'], axis=1)

cols = ['FTI', 'T4U', 'T3', 'TT4', 'TSH']
for col in cols:
        df[col] = df[col].replace('?', np.nan)
        df[col] = df[col].astype(float)
        df[col] = df[col].fillna(df[col].mean())

df.columns

label_encoders = {}
categorical_columns = ['on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
                       'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
                       'query_hyperthyroid', 'lithium', 'goitre','tumor','hypopituitary','psych'
                      ]

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(df['age'].mode()[0])

df['sex'] = df['sex'].replace({'F': 0, 'M': 1})

df['sex'] = df['sex'].replace('?', np.nan)
df = df.dropna(subset=['sex'])

df.isnull().sum()

if 'referral_source' in df.columns:
    dummies = pd.get_dummies(df['referral_source'], prefix='referral_source')
    df = pd.concat([df, dummies], axis=1)
    df = df.drop('referral_source', axis=1)
encoding = LabelEncoder()
df['Class'] = encoding.fit_transform(df['Class'])
df['Class'].value_counts()

df.head(5)

X = df.drop('Class', axis=1)
y = df['Class']

X
y
print(X.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True,random_state=42)
print(X_train.shape, X_test.shape)
print("X_train columns:", X_train.columns)
print("X_test columns:", X_test.columns)
oversampler = SMOTE(random_state=1)
X_train_smote, y_train_smote = oversampler.fit_resample(X_train, y_train)
class_distribution = pd.Series(y_train_smote).value_counts()
print(class_distribution)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
# X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'solver': ['liblinear', 'lbfgs', 'saga'],  # Different solvers to try
    'class_weight': ['balanced']  # Applying class weighting to handle imbalance
}

log_reg = LogisticRegression(random_state=42)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='precision')
grid_search.fit(X_train_smote, y_train_smote)

# Best hyperparameters
best_log_reg = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Predictions with probability threshold tuning
y_test_pred_proba = best_log_reg.predict_proba(X_test)[:, 1]

# Tune threshold (Increase to improve precision)
threshold = 0.85  # Increase the threshold to focus on higher precision
y_test_pred = (y_test_pred_proba >= threshold).astype(int)

# Training Accuracy
y_train_pred = best_log_reg.predict(X_train_smote)
train_accuracy = accuracy_score(y_train_smote, y_train_pred) * 100
test_accuracy = accuracy_score(y_test, y_test_pred) * 100
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))



# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
# Confusion Matrix Accuracy
conf_matrix_acc = np.trace(conf_matrix) / np.sum(conf_matrix)
print(f"Confusion Matrix Accuracy: {conf_matrix_acc:.5f}")

# Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse:.5f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.5f}")

# Precision, Recall, F1-Score
precision = precision_score(y_test, y_test_pred) * 100
recall = recall_score(y_test, y_test_pred) * 100
f1 = f1_score(y_test, y_test_pred) * 100
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=True, 
            xticklabels=['False', 'True'], yticklabels=['False', 'True'], linewidths=0.5)

plt.title('Confusion Matrix on Logistic Regression')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
