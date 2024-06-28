import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, \
    average_precision_score
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt


def plot_precision_recall(y_true, y_scores):
    precision = {}
    recall = {}
    average_precision = {}

    for i, class_label in enumerate(set(y_true)):
        precision[class_label], recall[class_label], _ = precision_recall_curve(y_true == class_label, y_scores[:, i])
        average_precision[class_label] = average_precision_score(y_true == class_label, y_scores[:, i])

    for class_label in precision:
        plt.plot(recall[class_label], precision[class_label], lw=2,
                 label='class {} (area = {:.2f})'.format(class_label, average_precision[class_label]))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("Precision-Recall Curve")
    plt.show()


def train_model(training_data_path):
    data = pd.read_csv(training_data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, stratify=data['label'], random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    min_class_count = y_train.value_counts().min()

    smote = SMOTE(random_state=42, k_neighbors=min(min_class_count - 1, 5))
    X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    plot_precision_recall(y_test, y_pred_proba)

    return model, vectorizer


if __name__ == "__main__":
    train_model('../data/training_data.csv')
