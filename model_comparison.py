# # Step 1: Import necessary libraries
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score, recall_score, f1_score,mean_squared_error
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Step 2: Load the dataset
# df = pd.read_csv('/dataset_thyroid_sick.csv')

# # Step 3: Handle missing values
# # Fill missing numerical values with the mean
# numeric_cols = df.select_dtypes(include=[np.number]).columns
# df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# # Fill missing categorical values with the mode (most frequent value)
# categorical_cols = df.select_dtypes(include=[object]).columns
# df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# # Verify there are no more missing values
# print("Missing values after imputation:")
# print(df.isnull().sum())

# # Step 4: Encode all categorical variables
# label_encoder = LabelEncoder()
# for col in df.select_dtypes(include=['object']).columns:
#     df[col] = label_encoder.fit_transform(df[col])

# # Verify the data types of the DataFrame
# print("Data types after encoding:")
# print(df.dtypes)

# # Step 5: Features (X) and Target (y)
# X = df.drop(columns=['target'])  # Ensure 'target' is the actual name of your target column
# y = df['target']

# # Check the shape of X and y
# print("Shape of X:", X.shape)
# print("Shape of y:", y.shape)

# # Step 6: Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Step 7: Scale the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # param_grid = {
# #     'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
# #     'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Regularization type
# #     'solver': ['liblinear', 'saga']  # Solvers that support L1 and Elastic Net
# # }

# # grid_search = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
# # grid_search.fit(X_train_scaled, y_train)

# # # Get the best parameters
# # best_params = grid_search.best_params_
# # print(f"Best parameters from GridSearch: {best_params}")

# # # Step 9: Initialize and train the Logistic Regression model with the best parameters
# # best_model = LogisticRegression(**best_params, max_iter=10000)
# # best_model.fit(X_train_scaled, y_train)
# # Step 8: Initialize and train the Logistic Regression model
# model = LogisticRegression()
# model.fit(X_train_scaled, y_train)

# # Step 9: Make predictions
# y_pred = model.predict(X_test_scaled)

# # Step 10: Evaluate the model
# accuracy = accuracy_score(y_test, y_pred) * 100
# print(f'Test Accuracy: {accuracy:.2f}%')
# y_train_pred = model.predict(X_train_scaled)
# accuracy_train = accuracy_score(y_train, y_train_pred) * 100
# print(f'Training Accuracy: {accuracy_train:.2f}%')
# precision = precision_score(y_test, y_pred, average='weighted') * 100
# recall = recall_score(y_test, y_pred, average='weighted') * 100
# f1 = f1_score(y_test, y_pred, average='weighted') * 100
# # print(f"Test Accuracy: {test_accuracy:.2f}%")
# print(f"Precision: {precision:.2f}%")
# print(f"Recall: {recall:.2f}%")
# print(f"F1-Score: {f1:.2f}%")

# # Step 12: Calculate Confusion Matrix Accuracy
# conf_matrix = confusion_matrix(y_test, y_pred)
# conf_matrix_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix) * 100  # Overall accuracy from confusion matrix
# print(f'Confusion Matrix Accuracy: {conf_matrix_accuracy:.2f}%')

# # Step 13: Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# print(f'Mean Squared Error (MSE): {mse:.4f}')
# print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
# # Step 11: Generate classification report and confusion matrix
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(conf_matrix)

# # Plotting the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score, recall_score, f1_score,mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
    df = df.drop(['TSH_measured','T3_measured','TT4_measured', 'T4U_measured','FTI_measured','TBG_measured','TBG'],axis=1)
    
    df['age'] = df['age'].fillna(df['age'].mode())
    col = ['FTI','T4U','T3','TT4','TSH']
    for i in col:
        df[i] = df[i].fillna(df[i].mean())
    df['sex'] = df['sex'].replace({'F':0,
                                  'M':1})
    df['sex'] = df['sex'].replace('?',np.nan)
    df = df.dropna()
    df = df.replace({'f':0,'t':1})
    df['Class'] = df['Class'].replace({'negative':0,'sick':1})
    dummies = pd.get_dummies(df['referral_source'])
    df = pd.concat([df,dummies],axis=1)
    df = df.drop('referral_source',axis=1)
    
    X = df.drop('Class',axis = 1)
    y = df['Class']
    print(X)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)
    
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = preprocess_inputs(data)

print(X_train)
# X.loc[X['sex']=='?']


oversampler = SMOTE(random_state=1)
X_train_smote, y_train_smote = oversampler.fit_resample(X_train, y_train)
print(X_train.shape)

models = {
    "                   Logistic Regression": LogisticRegression(),
   
    "                         Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine (Linear Kernel)": LinearSVC(),
    "   Support Vector Machine (RBF Kernel)": SVC(),
    "                        Neural Network": MLPClassifier(),
    "                         Random Forest": RandomForestClassifier(),
   
}


for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    print(name + " trained.")
                   
for name, model in models.items():
    print(name + ": {:.2f}%".format(model.score(X_test, y_test) * 100))        
# Hyperparameter tuning with class_weight balanced
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
threshold = 0.9  # Increase the threshold to focus on higher precision
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


# log_reg = LogisticRegression(class_weight='balanced')
# log_reg.fit(X_train_smote, y_train_smote)

# # Predictions
# y_train_pred = log_reg.predict(X_train_smote)
# y_test_pred = log_reg.predict(X_test)
# print(X_test)
# # Training and Test Accuracy
# train_accuracy = accuracy_score(y_train_smote, y_train_pred) * 100
# test_accuracy = accuracy_score(y_test, y_test_pred) * 100
# print(f"Training Accuracy: {train_accuracy:.2f}%")
# print(f"Test Accuracy: {test_accuracy:.2f}%")

# precision = precision_score(y_test, y_test_pred) * 100
# recall = recall_score(y_test, y_test_pred, average='weighted') * 100
# f1 = f1_score(y_test, y_test_pred, average='weighted') * 100
# print(f"Precision: {precision:.2f}%")
# print(f"Recall: {recall:.2f}%")
# print(f"F1-Score: {f1:.2f}%")
# # Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_test_pred))

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_test_pred)
# print("Confusion Matrix:")
# print(conf_matrix)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=['False', 'True'], yticklabels=['False', 'True'], linewidths=0.5)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
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

# label_encoder = LabelEncoder()
# for col in data.select_dtypes(include=['object']).columns:
#      data[col] = label_encoder.fit_transform(data[col])
     
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# # # Step 8: Initialize and train the Logistic Regression model
# model = LogisticRegression()
# model.fit(X_train_scaled, y_train)

# # # Step 9: Make predictions
# y_pred = model.predict(X_test_scaled)

# # # Step 10: Evaluate the model
# accuracy = accuracy_score(y_test, y_pred) * 100
# print(f'Test Accuracy: {accuracy:.2f}%')
# y_train_pred = model.predict(X_train_scaled)
# accuracy_train = accuracy_score(y_train, y_train_pred) * 100
# print(f'Training Accuracy: {accuracy_train:.2f}%')
# precision = precision_score(y_test, y_pred, average='weighted') * 100
# recall = recall_score(y_test, y_pred, average='weighted') * 100
# f1 = f1_score(y_test, y_pred, average='weighted') * 100

# print(f"Precision: {precision:.2f}%")
# print(f"Recall: {recall:.2f}%")
# print(f"F1-Score: {f1:.2f}%")

# # Step 12: Calculate Confusion Matrix Accuracy
# conf_matrix = confusion_matrix(y_test, y_pred)
# conf_matrix_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix) * 100  # Overall accuracy from confusion matrix
# print(f'Confusion Matrix Accuracy: {conf_matrix_accuracy:.2f}%')

# # Step 13: Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# print(f'Mean Squared Error (MSE): {mse:.4f}')
# print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
# # Step 11: Generate classification report and confusion matrix
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(conf_matrix)

# # Plotting the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show() 