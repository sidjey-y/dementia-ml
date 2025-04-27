"""
Simple Dementia Prediction Script

This script analyzes the dementia dataset to predict dementia status using
logistic regression and random forest classification.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

try:
    print("Loading dataset...")
    df = pd.read_csv('dementia_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    print("\nData Preprocessing...")
    df_cleaned = df.drop(['Subject ID', 'MRI ID'], axis=1)
    
    df_cleaned['Dementia_Status'] = df_cleaned['Group'].apply(lambda x: 1 if x in ['Demented', 'Converted'] else 0)
    
    print("\nCount of records by dementia status:")
    print(df_cleaned['Dementia_Status'].value_counts())
    print("\nPercentage of records by dementia status:")
    print(df_cleaned['Dementia_Status'].value_counts(normalize=True) * 100)
    
    missing_values = df_cleaned.isnull().sum()
    print("\nColumns with missing values:")
    print(missing_values[missing_values > 0])
    
    numerical_features = ['Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF'] 
    
    imputer = SimpleImputer(strategy='median')
    df_cleaned[numerical_features] = imputer.fit_transform(df_cleaned[numerical_features])
    
    df_cleaned['Gender_encoded'] = df_cleaned['M/F'].map({'M': 1, 'F': 0})
    df_cleaned['Hand_encoded'] = df_cleaned['Hand'].map({'R': 1, 'L': 0})
    
    features = ['Age', 'Gender_encoded', 'Hand_encoded', 'EDUC', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    X = df_cleaned[features]
    y = df_cleaned['Dementia_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    plt.figure(figsize=(8, 5))
    df_cleaned['Group'].value_counts().plot(kind='bar')
    plt.title('Distribution of Dementia Status')
    plt.tight_layout()
    plt.savefig('dementia_status_distribution.png')
    plt.close()
    
    print("\nTraining Logistic Regression model...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    precision_lr = precision_score(y_test, y_pred_lr)
    recall_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    roc_auc_lr = roc_auc_score(y_test, y_prob_lr)
    
    print("\nLogistic Regression Results:")
    print(f"Accuracy: {accuracy_lr:.4f}")
    print(f"Precision: {precision_lr:.4f}")
    print(f"Recall: {recall_lr:.4f}")
    print(f"F1 Score: {f1_lr:.4f}")
    print(f"ROC-AUC: {roc_auc_lr:.4f}")
    
    plt.figure(figsize=(6, 5))
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    plt.imshow(cm_lr, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.colorbar()
    plt.xticks([0, 1], ['Non-demented', 'Demented'])
    plt.yticks([0, 1], ['Non-demented', 'Demented'])
    
    thresh = cm_lr.max() / 2
    for i in range(cm_lr.shape[0]):
        for j in range(cm_lr.shape[1]):
            plt.text(j, i, str(cm_lr[i, j]),
                    horizontalalignment="center",
                    color="white" if cm_lr[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_lr.png')
    plt.close()
    
    print("\nTraining Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    y_pred_rf = rf.predict(X_test_scaled)
    y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
    
    print("\nRandom Forest Results:")
    print(f"Accuracy: {accuracy_rf:.4f}")
    print(f"Precision: {precision_rf:.4f}")
    print(f"Recall: {recall_rf:.4f}")
    print(f"F1 Score: {f1_rf:.4f}")
    print(f"ROC-AUC: {roc_auc_rf:.4f}")
    
    plt.figure(figsize=(6, 5))
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.imshow(cm_rf, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Random Forest')
    plt.colorbar()
    plt.xticks([0, 1], ['Non-demented', 'Demented'])
    plt.yticks([0, 1], ['Non-demented', 'Demented'])
    
    thresh = cm_rf.max() / 2
    for i in range(cm_rf.shape[0]):
        for j in range(cm_rf.shape[1]):
            plt.text(j, i, str(cm_rf[i, j]),
                    horizontalalignment="center",
                    color="white" if cm_rf[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_rf.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title('Feature Importances - Random Forest')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
    
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [accuracy_lr, accuracy_rf],
        'Precision': [precision_lr, precision_rf],
        'Recall': [recall_lr, recall_rf],
        'F1 Score': [f1_lr, f1_rf],
        'ROC-AUC': [roc_auc_lr, roc_auc_rf]
    })
    print(comparison_df.sort_values('ROC-AUC', ascending=False))
    
    best_model = 'Random Forest' if roc_auc_rf > roc_auc_lr else 'Logistic Regression'
    print(f"\nRecommendation: Based on ROC-AUC score, the {best_model} model is recommended for predicting dementia status.")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    traceback.print_exc() 