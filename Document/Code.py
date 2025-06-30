code.Py
# Revolutionizing Liver Care: Predicting Liver Cirrhosis Using ML

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# STEP 1: Load the dataset
# Replace this with your dataset path or URL
df = pd.read_csv('cirrhosis_dataset.csv')

# STEP 2: Data inspection and preprocessing
print("Initial Data Shape:", df.shape)
print(df.head())

# Drop rows with excessive missing values
df.dropna(thresh=int(0.9 * len(df.columns)), axis=0, inplace=True)

# Fill missing values with median (or use KNN imputation if desired)
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# STEP 3: Feature selection
X = df.drop(['Stage'], axis=1, errors='ignore')  # 'Stage' is the label in many cirrhosis datasets
y = df['Stage'] if 'Stage' in df.columns else df.iloc[:, -1]  # fallback if column is unnamed

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 4: Train/test split
X_train, X_test, y_train, y
