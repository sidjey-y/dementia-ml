"""
Support Vector Machine Model for Dementia Prediction

This script implements a Support Vector Machine (SVM) classifier to predict 
dementia status using the OASIS dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

try:
    print("Loading dataset...")
    df = pd.read_csv('dementia_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    
    print("\nData Preprocessing...")
    df_cleaned = df.drop(['Subject ID', 'MRI ID'], axis=1)
    
    df_cleaned['Dementia_Status'] = df_cleaned['Group'].apply(lambda x: 1 if x in ['Demented', 'Converted'] else 0)
    
    print("Count of records by dementia status:")
    print(df_cleaned['Dementia_Status'].value_counts())
    print("Percentage of records by dementia status:")
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
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    print("\nTraining SVM model with hyperparameter tuning...")
    
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1],
        'svm__kernel': ['rbf', 'linear'],
        'svm__class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\nBest Parameters for SVM:")
    print(grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    best_svm = grid_search.best_estimator_
    
    y_pred = best_svm.predict(X_test)
    y_prob = best_svm.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\nSVM Model Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Support Vector Machine')
    plt.colorbar()
    plt.xticks([0, 1], ['Non-demented', 'Demented'])
    plt.yticks([0, 1], ['Non-demented', 'Demented'])
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_svm.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_svm.png')
    plt.close()
    
    if 'linear' in best_svm.named_steps['svm'].kernel:
        importance = np.abs(best_svm.named_steps['svm'].coef_[0])
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]
        plt.title('Feature Importance - SVM (Linear Kernel)')
        plt.bar(range(X.shape[1]), importance[indices], align='center')
        plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance_svm.png')
        plt.close()
    
    print("\nSVM Model trained and evaluated successfully!")
    print("Visualizations saved: confusion_matrix_svm.png, roc_curve_svm.png")
    
    try:
        from simple_dementia_prediction import lr_results, rf_results
        
        comparison_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
            'Accuracy': [lr_results['accuracy'], rf_results['accuracy'], accuracy],
            'Precision': [lr_results['precision'], rf_results['precision'], precision],
            'Recall': [lr_results['recall'], rf_results['recall'], recall],
            'F1 Score': [lr_results['f1'], rf_results['f1'], f1],
            'ROC-AUC': [lr_results['roc_auc'], rf_results['roc_auc'], roc_auc]
        })
        
        print("\nModel Comparison:")
        print(comparison_df.sort_values('ROC-AUC', ascending=False))
    except:
        print("\nNote: Could not compare with other models. Run simple_dementia_prediction.py first to enable comparison.")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    traceback.print_exc() 