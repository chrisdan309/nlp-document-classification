from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ttest_rel
import numpy as np

def evaluate_model(y_true, y_pred, model_name, method):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"Resultados para {model_name} con {method}:")
    print(f"Precisión: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")
    return precision, recall, f1

def cross_validate_model(X, y, classifier):
    skf = StratifiedKFold(n_splits=5)
    f1_scores = cross_val_score(classifier, X, y, cv=skf, scoring='f1_weighted')
    return f1_scores

def perform_statistical_test(f1_scores_method1, f1_scores_method2, method1_name, method2_name):
    t_statistic, p_value = ttest_rel(f1_scores_method1, f1_scores_method2)
    print(f"Prueba estadística entre {method1_name} y {method2_name}:")
    print(f"T-statistic: {t_statistic:.4f}, P-value: {p_value:.4f}\n")
    return t_statistic, p_value
