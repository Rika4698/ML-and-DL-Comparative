import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
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
    df = df.drop(['TSH_measured','T3_measured','TT4_measured', 'T4U_measured','FTI_measured','TBG_measured','TBG'], axis=1)
    
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
# SMOTE for class balancing
oversampler = SMOTE(random_state=1)
X_train_smote, y_train_smote = oversampler.fit_resample(X_train, y_train)

# # Define models and their parameters
# models = {
#     "Logistic Regression": {
#         "model": LogisticRegression(random_state=42),
#         "param_grid": {
#             'C': [0.01, 0.1, 1, 10, 100],
#             'solver': ['liblinear', 'lbfgs', 'saga'],
#             'class_weight': ['balanced']
#         },
#         "threshold": 0.9
#     },
#     "Decision Tree": {
#         "model": DecisionTreeClassifier(random_state=42),
#         "param_grid": {
#             'max_depth': [3, 5, 7, 10],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4]
#         },
#         "threshold": 0.67
#     },
#     "Support Vector Machine (Linear Kernel)": {
#         "model": SVC(probability=True, random_state=42),  # Use SVC instead of LinearSVC
#         "param_grid": {
#             'C': [0.01, 0.1, 1, 10, 100],
#             'kernel': ['linear'],  # Set the kernel to linear
#             'class_weight': ['balanced']
#         },
#         "threshold": 0.9
#     },
#     "Support Vector Machine (RBF Kernel)": {
#         "model": SVC(probability=True, random_state=42),
#         "param_grid": {
#             'C': [0.01, 0.1, 1, 10, 100],
#             'kernel': ['rbf'],
#             'class_weight': ['balanced']
#         },
#         "threshold": 0.79
#     },
#     "Random Forest": {
#         "model": RandomForestClassifier(random_state=42),
#         "param_grid": {},
#         "threshold": 0.59
#     }
# }
# # Fit each model, tune hyperparameters, and evaluate
# for name, model_info in models.items():
#     model = model_info["model"]
#     param_grid = model_info["param_grid"]
#     threshold = model_info["threshold"]
    
#     # Perform hyperparameter tuning using GridSearchCV
#     grid_search = GridSearchCV(model, param_grid, cv=5, scoring='precision', n_jobs=-1)
#     grid_search.fit(X_train_smote, y_train_smote)

#     # Best estimator after GridSearch
#     best_model = grid_search.best_estimator_
#     print(f"Training {name}...")
#     print(f"Best Hyperparameters for {name}: {grid_search.best_params_}")
    
#     # Fit the model with the training data
#     best_model.fit(X_train_smote, y_train_smote)
    
#     # Predict the probabilities
#     if hasattr(best_model, "predict_proba"):  # Ensure the model supports predict_proba
#         y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
#         # Apply threshold tuning
#         y_test_pred = (y_test_pred_proba >= threshold).astype(int)
#     else:
#         # Use decision function for models like SVM without probability prediction
#         y_test_pred_decision = best_model.decision_function(X_test)
#         y_test_pred = (y_test_pred_decision >= threshold).astype(int)

    
#     train_accuracy = accuracy_score(y_train_smote, best_model.predict(X_train_smote)) * 100
#     test_accuracy = accuracy_score(y_test, y_test_pred) * 100
#     print(f"Training Accuracy: {train_accuracy:.2f}%")
#     print(f"Test Accuracy: {test_accuracy:.2f}%")

#     # Classification Report
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_test_pred))
#     precision = precision_score(y_test, y_test_pred) * 100
#     recall = recall_score(y_test, y_test_pred) * 100
#     f1 = f1_score(y_test, y_test_pred) * 100
    
#     print(f"\n{name}:")
#     print(f"Training Accuracy: {train_accuracy:.2f}%")
#     print(f"Test Accuracy: {test_accuracy:.2f}%")
#     print(f"Precision: {precision:.2f}%")
#     print(f"Recall: {recall:.2f}%")
#     print(f"F1-Score: {f1:.2f}%")

#     # Confusion Matrix
#     conf_matrix = confusion_matrix(y_test, y_test_pred)
#     conf_matrix_acc = np.trace(conf_matrix) / np.sum(conf_matrix) * 100
#     print(f"Confusion Matrix Accuracy: {conf_matrix_acc:.2f}%")
    
#     # Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
#     mse = mean_squared_error(y_test, y_test_pred)
#     rmse = np.sqrt(mse)
#     print(f"Mean Squared Error (MSE): {mse:.2f}")
#     print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

#     # Plot the confusion matrix
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, 
#                 xticklabels=['False', 'True'], yticklabels=['False', 'True'], linewidths=0.5)
#     plt.title(f'Confusion Matrix on {name}')
#     plt.xlabel('Predicted label')
#     plt.ylabel('True label')
#     plt.show()
# Data from the model evaluation
labels = ['Test Accuracy (%)', 'Training Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
logistic_regression = [96.69, 91.19, 77.97, 66.67, 71.88]
decision_tree = [98.44, 99.73, 88.24, 86.96, 87.59]
svm_linear = [96.78, 91.61, 80.36, 65.22, 72.00]
svm_rbf = [97.15, 99.33, 82.76, 69.57, 75.59]
random_forest = [98.25, 100.00, 87.88, 84.06, 85.93]

# # Bar width and space settings
# barWidth = 0.1
# space_between_bars = 0.1  # Additional space between each group of bars
# value_space = 1  # Space above the bars for value labels
# height_factor = 0.8  # Reduce the height of the bars

# # Set position of bars on X axis with added space
# r1 = np.arange(len(labels))
# r2 = [x + barWidth + space_between_bars for x in r1]
# r3 = [x + barWidth + space_between_bars for x in r2]
# r4 = [x + barWidth + space_between_bars for x in r3]
# r5 = [x + barWidth + space_between_bars for x in r4]

# Bar width and space settings
barWidth = 0.1
space_between_bars = 0.05  # Additional space between each group of bars
value_space = 2  # Space above the bars for value labels
# height_factor = 0.5  # Reduce the height of the bars

# Set position of bars on X axis with added space
r1 = np.arange(len(labels))
r2 = [x + barWidth + space_between_bars for x in r1]
r3 = [x + barWidth + space_between_bars for x in r2]
r4 = [x + barWidth + space_between_bars for x in r3]
r5 = [x + barWidth + space_between_bars for x in r4]

# Create the bar chart
plt.figure(figsize=(10, 6))
bars1 = plt.bar(r1, logistic_regression, color='lightblue', width=barWidth, edgecolor='grey', label='Logistic Regression')
bars2 = plt.bar(r2, decision_tree, color='cornflowerblue', width=barWidth, edgecolor='grey', label='Decision Tree')
bars3 = plt.bar(r3, svm_linear, color='gray', width=barWidth, edgecolor='grey', label='SVM (Linear Kernel)')
bars4 = plt.bar(r4, svm_rbf, color='lightgreen', width=barWidth, edgecolor='grey', label='SVM (RBF Kernel)')
bars5 = plt.bar(r5, random_forest, color='green', width=barWidth, edgecolor='grey', label='Random Forest')

# Add labels and title
# plt.xlabel('Metrics', fontweight='bold')
plt.xticks([r + 2 * (barWidth + space_between_bars) for r in range(len(labels))], labels, fontsize=10, rotation=0, ha='center')
plt.title('RESULTS OF ML ALGORITHMS', fontsize=20,pad=58,fontweight='bold', color='grey')

# Add values inside the bars with additional spacing
def add_value_labels(bars, value_space=2):
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + value_space, f'{yval:.2f}%', ha='center', va='bottom',
                 fontsize=10, rotation=90, color='gray', fontweight='bold')

# Apply the function to all bars with added space for value labels
add_value_labels(bars1, value_space)
add_value_labels(bars2, value_space)
add_value_labels(bars3, value_space)
add_value_labels(bars4, value_space)
add_value_labels(bars5, value_space)

# Remove top and right spines (borders)
ax = plt.gca()  # Get current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside the graph area

# Adjust margins to remove space above and on the right side
plt.subplots_adjust(top=0.7, bottom=0.15, left=0.1, right=0.95, hspace=0.4, wspace=0.4)

# Show the chart
plt.tight_layout()
plt.show()