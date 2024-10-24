# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score, recall_score, f1_score,mean_squared_error
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import LinearSVC, SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.ensemble import BaggingClassifier
# import matplotlib.pyplot as plt
# from imblearn.over_sampling import RandomOverSampler, SMOTE
# import tensorflow as tf
# from tensorflow.keras.models import Sequential # type: ignore
# from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
# from tensorflow.keras.utils import to_categorical
# # from tensorflow.keras.regularizers import l2
# from keras_tuner import RandomSearch
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')

# data = pd.read_csv('dataset_thyroid_sick.csv')
# # pd.set_option('max_columns',None)
# print(data)
# print(data.describe())
# print(data.info())
# print(data.isna().sum())

# print(data.dtypes)
# for col in data.columns:
#     if data[col].dtypes == 'object':
#         print(col, data[col].unique())

# data['age'] = data['age'].apply(pd.to_numeric, errors='coerce')
# col = ['FTI','T4U','T3','TT4','TSH']
# for columns in col:
#         data[columns] = data[columns].apply(pd.to_numeric, errors='coerce')
# print(data.dtypes)
# for col in data.columns:
#     if data[col].dtypes == 'object':
#         print(col, data[col].unique())

# def preprocess_inputs(df):
#     df = df.copy()
#     df = df.drop(['TSH_measured','T3_measured','TT4_measured', 'T4U_measured','FTI_measured','TBG_measured','TBG'],axis=1)
    
#     df['age'] = df['age'].fillna(df['age'].mode())
#     col = ['FTI','T4U','T3','TT4','TSH']
#     for i in col:
#         df[i] = df[i].fillna(df[i].mean())
#     df['sex'] = df['sex'].replace({'F':0,
#                                   'M':1})
#     df['sex'] = df['sex'].replace('?',np.nan)
#     df = df.dropna()
#     df = df.replace({'f':0,'t':1})
#     df['Class'] = df['Class'].replace({'negative':0,'sick':1})
#     dummies = pd.get_dummies(df['referral_source'])
#     df = pd.concat([df,dummies],axis=1)
#     df = df.drop('referral_source',axis=1)
    
#     X = df.drop('Class',axis = 1)
#     y = df['Class']
#     print(X.dtypes)
#     print(y.dtypes)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)
    
#     scaler = StandardScaler()
#     scaler.fit(X_train)
    
#     X_train = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
#     X_test = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
#     return X_train, X_test, y_train, y_test
# X_train, X_test, y_train, y_test = preprocess_inputs(data)

# print(X_train)
# # X.loc[X['sex']=='?']
# oversampler = SMOTE(random_state=1)
# X_train_smote, y_train_smote = oversampler.fit_resample(X_train, y_train)
# class_distribution = pd.Series(y_train_smote).value_counts()
# print(class_distribution)
# models = {
#     "                   Logistic Regression": LogisticRegression(),
#     "                   K-Nearest Neighbors": KNeighborsClassifier(),
#     "                         Decision Tree": DecisionTreeClassifier(),
#     "Support Vector Machine (Linear Kernel)": LinearSVC(),
#     "   Support Vector Machine (RBF Kernel)": SVC(),
#     "                        Neural Network": MLPClassifier(),
#     "                         Random Forest": RandomForestClassifier(),
#     "                     Gradient Boosting": GradientBoostingClassifier(),
#     "                    Bagging Classifier": BaggingClassifier()
# }


# for name, model in models.items():
#     model.fit(X_train_smote, y_train_smote)
#     print(name + " trained.")
                   
# for name, model in models.items():
#     print(name + ": {:.2f}%".format(model.score(X_test, y_test) * 100))   

# # Reshape the data for RNN
# X_train_rnn = X_train_smote.values.reshape(X_train_smote.shape[0], X_train_smote.shape[1], 1)
# X_test_rnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Convert labels to categorical
# y_train_rnn = to_categorical(y_train_smote)
# y_test_rnn = to_categorical(y_test)

# # Build the model using Keras Tuner
# def build_model(hp):
#     model = Sequential()
    
#     # Tune the number of units in the LSTM layer
#     model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32), input_shape=(X_train_rnn.shape[1], 1), return_sequences=True))
    
#     # Tune the dropout rate
#     model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    
#     model.add(LSTM(50))
#     model.add(Dropout(0.2))
    
#     # Output layer
#     model.add(Dense(2, activation='softmax'))
    
#     # Compile the model
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
#     return model

# # Initialize the Keras Tuner with Random Search
# tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5, executions_per_trial=3)

# # Search for the best hyperparameters
# tuner.search(X_train_rnn, y_train_rnn, epochs=40, validation_split=0.1)

# # Get the best model after hyperparameter tuning
# best_model = tuner.get_best_models(num_models=1)[0]

# history = best_model.fit(X_train_rnn, y_train_rnn, epochs=40, batch_size=32, validation_split=0.1)

# # Evaluate the model on the test set
# # Get model predictions (probabilities)
# y_pred_prob = best_model.predict(X_test_rnn)

# # Adjust the threshold (e.g., 0.6 instead of 0.5)
# threshold = 0.8
# y_pred_classes = (y_pred_prob[:, 1] > threshold).astype(int)
# y_test_classes = np.argmax(y_test_rnn, axis=1)

# # Train Accuracy
# train_accuracy = best_model.evaluate(X_train_rnn, y_train_rnn, verbose=0)[1] * 100
# print(f'Training Accuracy: {train_accuracy:.2f}%')

# # Test Accuracy
# test_loss, test_accuracy = best_model.evaluate(X_test_rnn, y_test_rnn, verbose=0)
# test_accuracy = test_accuracy * 100
# print(f'Test Accuracy: {test_accuracy:.2f}%')

# # Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test_classes, y_pred_classes))

# # Precision, Recall, F1-Score
# precision = precision_score(y_test_classes, y_pred_classes) * 100
# recall = recall_score(y_test_classes, y_pred_classes) * 100
# f1 = f1_score(y_test_classes, y_pred_classes) * 100
# print(f'Precision: {precision:.2f}%')
# print(f'Recall: {recall:.2f}%')
# print(f'F1-Score: {f1:.2f}%')

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.title("Confusion Matrix")
# plt.ylabel('Actual Class')
# plt.xlabel('Predicted Class')
# plt.show()



# # Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
# mse = mean_squared_error(y_test_classes, y_pred_classes)
# rmse = np.sqrt(mse)
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')


# X_train_rnn = X_train_smote.values.reshape(X_train_smote.shape[0], X_train_smote.shape[1], 1)
# X_test_rnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Convert labels to categorical
# y_train_rnn = to_categorical(y_train_smote)
# y_test_rnn = to_categorical(y_test)

# # Build the RNN model
# model = Sequential()
# model.add(LSTM(50, input_shape=(X_train_rnn.shape[1], 1), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(50))
# model.add(Dropout(0.2))
# model.add(Dense(2, activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train_rnn, y_train_rnn, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# # Evaluate the model
# y_pred_rnn = model.predict(X_test_rnn)
# y_pred_classes = np.argmax(y_pred_rnn, axis=1)

# # Metrics
# accuracy = accuracy_score(y_test, y_pred_classes)
# precision = precision_score(y_test, y_pred_classes)
# recall = recall_score(y_test, y_pred_classes)
# f1 = f1_score(y_test, y_pred_classes)
# mse = mean_squared_error(y_test, y_pred_classes)
# rmse = np.sqrt(mse)

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred_classes)

# # Print results
# print(f'Accuracy: {accuracy * 100:.2f}%')
# print(f'Precision: {precision * 100:.2f}%')
# print(f'Recall: {recall * 100:.2f}%')
# print(f'F1 Score: {f1 * 100:.2f}%')
# print(f'Mean Squared Error: {mse:.4f}')
# print(f'Root Mean Squared Error: {rmse:.4f}')

# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score, recall_score, f1_score,mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.regularizers import l2
from keras_tuner import RandomSearch
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
    print(X.dtypes)
    print(y.dtypes)
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
class_distribution = pd.Series(y_train_smote).value_counts()
print(class_distribution)
models = {
    "                   Logistic Regression": LogisticRegression(),
    "                   K-Nearest Neighbors": KNeighborsClassifier(),
    "                         Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine (Linear Kernel)": LinearSVC(),
    "   Support Vector Machine (RBF Kernel)": SVC(),
    "                        Neural Network": MLPClassifier(),
    "                         Random Forest": RandomForestClassifier(),
    "                     Gradient Boosting": GradientBoostingClassifier(),
    "                    Bagging Classifier": BaggingClassifier()
}


for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    print(name + " trained.")
                   
for name, model in models.items():
    print(name + ": {:.2f}%".format(model.score(X_test, y_test) * 100))   


# Check the shape of the data
print("Shape of X_train_smote:", X_train_smote.shape)
print("Shape of X_test:", X_test.shape)

# Number of features (columns) in the dataset
# n_features = X_train_smote.shape[1]

# Reshape the data for RNN
X_train_rnn = X_train_smote.values.reshape(X_train_smote.shape[0], X_train_smote.shape[1], 1)
X_test_rnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert labels to categorical
y_train_rnn = to_categorical(y_train_smote)
y_test_rnn = to_categorical(y_test)

# Build the model using Keras Tuner
def build_model(hp):
    model = Sequential()
    
    # Increase the number of LSTM layers and units
    model.add(LSTM(units=hp.Int('units', min_value=63, max_value=256, step=64), input_shape=(X_train_rnn.shape[1], 1), return_sequences=True))
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Add a second LSTM layer
    model.add(LSTM(units=hp.Int('units2', min_value=63, max_value=256, step=64)))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(2, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5, executions_per_trial=3)

# Search for the best hyperparameters
tuner.search(X_train_rnn, y_train_rnn, epochs=40, validation_split=0.1)

# Get the best model after hyperparameter tuning
best_model = tuner.get_best_models(num_models=1)[0]

history = best_model.fit(X_train_rnn, y_train_rnn, epochs=40, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
# Get model predictions (probabilities)
y_pred_prob = best_model.predict(X_test_rnn)

# Adjust the threshold (e.g., 0.6 instead of 0.5)
threshold = 0.8
y_pred_classes = (y_pred_prob[:, 1] > threshold).astype(int)
y_test_classes = np.argmax(y_test_rnn, axis=1)


# Calculate and print accuracies
train_accuracy = best_model.evaluate(X_train_rnn, y_train_rnn, verbose=0)[1] * 100
print(f'Training Accuracy: {train_accuracy:.2f}%')

test_loss, test_accuracy = best_model.evaluate(X_test_rnn, y_test_rnn, verbose=0)
test_accuracy = test_accuracy * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Precision, Recall, F1-Score
precision = precision_score(y_test_classes, y_pred_classes) * 100
recall = recall_score(y_test_classes, y_pred_classes) * 100
f1 = f1_score(y_test_classes, y_pred_classes) * 100
print(f'Precision: {precision:.2f}%')
print(f'Recall: {recall:.2f}%')
print(f'F1-Score: {f1:.2f}%')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=True, 
            xticklabels=['False', 'True'], yticklabels=['False', 'True'], linewidths=0.5)
plt.title("Confusion Matrix")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()