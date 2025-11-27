"""
Premade training + evaluation functions for algorithms
used on different unstructured data types.

Each function:
- accepts (X, y) already preprocessed/feature-extracted
- splits into train/test
- trains the algorithm
- returns metrics: accuracy, precision, recall, f1, report, and the model
"""

from typing import Any, Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Text algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# Generic classical ML for multiple types (logs, sensor, etc.)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Sequence / time-series style (using classical ML on features)
from sklearn.ensemble import IsolationForest


# ========== Generic helpers ==========

def _split_data(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Any = None,
) -> Tuple[Any, Any, Any, Any]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def _evaluate_model(
    model,
    X_test,
    y_test,
    average: str = "weighted",
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average, zero_division=0)
    rec = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "classification_report": report,
        "y_pred": y_pred,
    }


def _train_and_eval(
    model,
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Any = None,
    average: str = "weighted",
) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = _split_data(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    model.fit(X_train, y_train)
    metrics = _evaluate_model(model, X_test, y_test, average=average)
    metrics["model"] = model
    metrics["X_train"] = X_train
    metrics["X_test"] = X_test
    metrics["y_train"] = y_train
    metrics["y_test"] = y_test
    return metrics


# ========== 1. TEXT DATA: emails, chats, posts, docs, transcripts ==========

def text_logistic_regression(X, y, **kwargs):
    """
    Text 
    - Emails, chat messages, social media posts, blogs, articles, PDFs, Word docs, books, transcripts.
    Algorithm:
    - Logistic Regression on BoW/TF-IDF/embeddings.
    """
    model = LogisticRegression(max_iter=1000)
    return _train_and_eval(model, X, y, **kwargs)


def text_linear_svm(X, y, **kwargs):
    """
    Text 
    - Emails, chat messages, social media posts, blogs, articles, docs, transcripts.
    Algorithm:
    - Linear Support Vector Machine (LinearSVC).
    """
    model = LinearSVC()
    return _train_and_eval(model, X, y, **kwargs)


def text_svm_rbf(X, y, **kwargs):
    """
    Text 
    - General text classification where non-linear decision boundary may help.
    Algorithm:
    - SVC with RBF kernel.
    """
    model = SVC(kernel="rbf", probability=True)
    return _train_and_eval(model, X, y, **kwargs)


def text_multinomial_nb(X, y, **kwargs):
    """
    Text 
    - Discrete word-count features: spam detection, sentiment, topic classification.
    Algorithm:
    - Multinomial Naive Bayes.
    """
    model = MultinomialNB()
    return _train_and_eval(model, X, y, **kwargs)


def text_bernoulli_nb(X, y, **kwargs):
    """
    Text 
    - Binary features (word present/absent), short messages, etc.
    Algorithm:
    - Bernoulli Naive Bayes.
    """
    model = BernoulliNB()
    return _train_and_eval(model, X, y, **kwargs)


def text_random_forest(X, y, **kwargs):
    """
    Text 
    - When using engineered features (TF-IDF, embeddings) and wanting non-linear trees.
    Algorithm:
    - Random Forest Classifier.
    """
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    return _train_and_eval(model, X, y, **kwargs)


# ========== 2. AUDIO DATA: voice, calls, podcasts, music, interviews ==========

def audio_svm_on_mfcc(X, y, **kwargs):
    """
    Audio 
    - Voice recordings, phone calls, podcasts, music, interviews.
    Expectation:
    - X contains MFCC/spectrogram features per clip.
    Algorithm:
    - SVM (RBF kernel) on audio features.
    """
    model = SVC(kernel="rbf", probability=True)
    return _train_and_eval(model, X, y, **kwargs)


def audio_random_forest_on_mfcc(X, y, **kwargs):
    """
    Audio 
    - Audio event/music genre/speaker classification using MFCC or other features.
    Algorithm:
    - Random Forest Classifier on extracted audio features.
    """
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    return _train_and_eval(model, X, y, **kwargs)


# ========== 3. IMAGE DATA: photos, medical, satellite, scanned docs, handwriting ==========

def image_knn_on_features(X, y, **kwargs):
    """
    Image 
    - Photographs, medical images, satellite images, scanned documents, handwritten notes.
    Expectation:
    - X contains precomputed feature vectors (e.g., CNN embeddings, SIFT/HOG, etc.).
    Algorithm:
    - k-Nearest Neighbors classifier on image features.
    """
    model = KNeighborsClassifier(n_neighbors=5)
    return _train_and_eval(model, X, y, **kwargs)


def image_random_forest_on_features(X, y, **kwargs):
    """
    Image 
    - Same as above, but using tree ensemble.
    Algorithm:
    - Random Forest Classifier on image feature vectors.
    """
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    return _train_and_eval(model, X, y, **kwargs)


# ========== 4. VIDEO DATA: movies, CCTV, lectures, YouTube, screen recordings ==========

def video_rf_on_frame_features(X, y, **kwargs):
    """
    Video 
    - Movies, surveillance footage, lectures, YouTube videos, screen recordings.
    Expectation:
    - X contains aggregated frame/clip-level feature vectors (e.g., pooled CNN features).
    Algorithm:
    - Random Forest Classifier on video feature vectors.
    """
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    return _train_and_eval(model, X, y, **kwargs)


def video_gb_on_frame_features(X, y, **kwargs):
    """
    Video 
    - Same video types, where gradient boosting may capture subtle differences.
    Algorithm:
    - Gradient Boosting Classifier on video features.
    """
    model = GradientBoostingClassifier(random_state=42)
    return _train_and_eval(model, X, y, **kwargs)


# ========== 5. SENSOR / IoT RAW DATA: accelerometer, health, GPS, raw signals ==========

def sensor_rf_on_features(X, y, **kwargs):
    """
    Sensor / IoT 
    - Accelerometer readings, smartwatch health signals, GPS traces, raw sensor signals.
    Expectation:
    - X contains time-series features (e.g., statistics, FFT, windowed features).
    Algorithm:
    - Random Forest Classifier for activity/condition classification.
    """
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    return _train_and_eval(model, X, y, **kwargs)


def sensor_isolation_forest(X, y=None, contamination: float = 0.05, **kwargs):
    """
    Sensor / IoT data (anomaly detection):
    - Detect anomalies in raw sensor streams, health data, etc.
    Algorithm:
    - Isolation Forest (unsupervised). Returns fitted model and anomaly scores.
    NOTE:
    - Metrics here are not accuracy/precision/recall because labels may be absent.
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    scores = model.decision_function(X)
    return {
        "model": model,
        "scores": scores,
    }


# ========== 6. WEB DATA: HTML pages, crawled content, online comments, social interactions ==========

def web_text_logistic_regression(X, y, **kwargs):
    """
    Web 
    - Web pages (after HTML parsing), crawled content, online comments.
    Expectation:
    - X is text features (TF-IDF/embeddings) from page/comment content.
    Algorithm:
    - Logistic Regression for classification (e.g., topic, toxicity, spam).
    """
    model = LogisticRegression(max_iter=1000)
    return _train_and_eval(model, X, y, **kwargs)


def web_text_linear_svm(X, y, **kwargs):
    """
    Web 
    - Same as above, good for high-dimensional sparse features.
    Algorithm:
    - Linear SVM (LinearSVC).
    """
    model = LinearSVC()
    return _train_and_eval(model, X, y, **kwargs)


# ========== 7. LOG DATA: system, application, server logs (unstructured) ==========

def log_text_logistic_regression(X, y, **kwargs):
    """
    Log 
    - System logs, application logs, server logs converted to text features.
    Algorithm:
    - Logistic Regression for log classification / anomaly vs normal labeling.
    """
    model = LogisticRegression(max_iter=1000)
    return _train_and_eval(model, X, y, **kwargs)


def log_text_random_forest(X, y, **kwargs):
    """
    Log 
    - Same log sources with richer non-linear patterns.
    Algorithm:
    - Random Forest Classifier on log text features.
    """
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    return _train_and_eval(model, X, y, **kwargs)


# ========== 8. OTHER UNSTRUCTURED: emails+attachments, noisy audio, mixed video, scanned bills ==========

def ocr_text_logistic_regression(X, y, **kwargs):
    """
    OCR text 
    - Scanned receipts or bills, scanned documents after OCR,
      text extracted from images or PDFs.
    Algorithm:
    - Logistic Regression on OCR text features (e.g., expense type classification).
    """
    model = LogisticRegression(max_iter=1000)
    return _train_and_eval(model, X, y, **kwargs)


def email_with_attachments_text_nb(X, y, **kwargs):
    """
    Email + attachments:
    - Emails whose body and attachment text are merged into a single text field.
    Algorithm:
    - Multinomial Naive Bayes for spam / intent / topic detection.
    """
    model = MultinomialNB()
    return _train_and_eval(model, X, y, **kwargs)


def noisy_audio_svm_on_mfcc(X, y, **kwargs):
    """
    Noisy audio:
    - Audio with background noise (calls, recordings).
    Expectation:
    - X has robust features (e.g., MFCC with noise reduction).
    Algorithm:
    - SVM classifier on audio features.
    """
    model = SVC(kernel="rbf", probability=True)
    return _train_and_eval(model, X, y, **kwargs)
