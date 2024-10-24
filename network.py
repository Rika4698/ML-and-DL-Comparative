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
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model = Sequential()
model.add(Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train_smote, epochs=40, batch_size=32, validation_split=0.2)

y_train_pred = (model.predict(X_train_scaled) > 0.6).astype(int)

y_test_pred = (model.predict(X_test_scaled) > 0.6).astype(int)

training_accuracy = accuracy_score(y_train_smote, y_train_pred)
print(f"Training accuracy for RNN: {training_accuracy * 100:.2f} %")

# Testing accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing accuracy for RNN: {test_accuracy * 100:.2f} %")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)
print('\n')

# Precision
precision = precision_score(y_test, y_test_pred)
print("Precision:")
print(precision)
print('\n')

#Recall
recall = recall_score(y_test, y_test_pred)
print("Recall:")
print(recall)
print('\n')

#F1 Score
f1 = f1_score(y_test, y_test_pred)
print("F1 Score:")
print(f1)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion Matrix")
plt.show()