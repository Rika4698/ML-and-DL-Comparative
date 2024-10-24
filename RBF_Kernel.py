import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('dataset_thyroid_sick.csv')
# pd.set_option('max_columns',None)
print(data)
print(data.describe())
print(data.info())
print(data.isna().sum())

print(data.dtypes)
for col in data.columns:
    if data[col].dtypes == 'object':
        print(col, data[col].unique())

data['age'] = data['age'].apply(pd.to_numeric, errors='coerce')
col = ['FTI','T4U','T3','TT4','TSH']
for columns in col:
        data[columns] = data[columns].apply(pd.to_numeric, errors='coerce')
print(data.dtypes)
for col in data.columns:
    if data[col].dtypes == 'object':
        print(col, data[col].unique())

def preprocess_inputs(df):
    df = df.copy()
    df = df.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'TBG'], axis=1)

    df['age'] = df['age'].fillna(df['age'].mode())
    col = ['FTI', 'T4U', 'T3', 'TT4', 'TSH']
    for i in col:
        df[i] = df[i].fillna(df[i].mean())

    df['sex'] = df['sex'].replace({'F': 0, 'M': 1})
    df['sex'] = df['sex'].replace('?', np.nan)
    df = df.dropna()
    df = df.replace({'f': 0, 't': 1})
    df['Class'] = df['Class'].replace({'negative': 0, 'sick': 1})

    dummies = pd.get_dummies(df['referral_source'])
    df = pd.concat([df, dummies], axis=1)
    df = df.drop('referral_source', axis=1)

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    return X_train, X_test, y_train, y_test

# Preprocess the inputs
X_train, X_test, y_train, y_test = preprocess_inputs(data)
print(X_train)
# X.loc[X['sex']=='?']
# Apply SMOTE for class balancing
oversampler = SMOTE(random_state=1)
X_train_smote, y_train_smote = oversampler.fit_resample(X_train, y_train)


svm_model = SVC()

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['rbf'],              # Using RBF kernel
    'class_weight': ['balanced']    # Applying class weighting to handle imbalance
}

grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='precision')
grid_search.fit(X_train_smote, y_train_smote)

# Best hyperparameters
best_svm_model = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Predictions with probability threshold tuning
y_test_pred_proba = best_svm_model.decision_function(X_test)  # Use decision function for SVM
threshold = 0.79  # Increase the threshold to focus on higher precision
y_test_pred = (y_test_pred_proba >= threshold).astype(int)
y_train_pred = best_svm_model.predict(X_train_smote)
# # SVM Model with RBF Kernel
# svm_rbf = SVC(kernel='rbf', probability=True)  # Added probability=True for probability predictions if needed
# svm_rbf.fit(X_train_smote, y_train_smote)

# # Predictions
# y_train_pred = svm_rbf.predict(X_train_smote)
# y_test_pred = svm_rbf.predict(X_test)

# Training and Test Accuracy
train_accuracy = accuracy_score(y_train_smote, y_train_pred) * 100
test_accuracy = accuracy_score(y_test, y_test_pred) * 100
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
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

# Confusion Matrix visualization
print("Confusion Matrix:")
print(conf_matrix)
# colors = ["skyblue", "lightpurple"]  # Start and end colors
# custom_cmap = LinearSegmentedColormap.from_list("custom_blue_purple", colors)
sns.heatmap(conf_matrix, annot=True, fmt='d',cmap="Blues",  cbar=True, 
            xticklabels=['False', 'True'], yticklabels=['False', 'True'], linewidths=0.5)

plt.title('Confusion Matrix on Support Vector Machine (RBF Kernel)')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_test_pred)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # Confusion Matrix Accuracy
# conf_matrix_acc = np.trace(conf_matrix) / np.sum(conf_matrix) * 100
# print(f"Confusion Matrix Accuracy: {conf_matrix_acc:.2f}%")

# # Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
# mse = mean_squared_error(y_test, y_test_pred)
# rmse = np.sqrt(mse)
# print(f"Mean Squared Error (MSE): {mse:.2f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# # Precision, Recall, F1-Score
# precision = precision_score(y_test, y_test_pred) * 100
# recall = recall_score(y_test, y_test_pred) * 100
# f1 = f1_score(y_test, y_test_pred) * 100
# print(f"Precision: {precision:.2f}%")
# print(f"Recall: {recall:.2f}%")
# print(f"F1-Score: {f1:.2f}%")