"""
Simple test script for unstructured_algorithms.py

- Creates fake numeric feature data (X, y)
- Calls several premade functions for different data types
- Prints accuracy, precision, recall, and F1
"""

import numpy as np

import ml_unstructured_models as ua


def print_metrics(name, metrics):
    print("=" * 60)
    print(f"{name}")
    print("-" * 60)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print()


def main():
    # ---------------- TEXT-LIKE TEST DATA ----------------
    # Simulate TF-IDF-like text features
    X_text = np.random.rand(200, 100)   # 200 samples, 100 features
    y_text = np.random.randint(0, 2, size=200)  # binary labels

    text_lr_metrics = ua.text_logistic_regression(X_text, y_text)
    print_metrics("TEXT - Logistic Regression", text_lr_metrics)

    text_svm_metrics = ua.text_linear_svm(X_text, y_text)
    print_metrics("TEXT - Linear SVM", text_svm_metrics)

    # ---------------- AUDIO-LIKE TEST DATA ----------------
    # Simulate MFCC-like audio features
    X_audio = np.random.rand(150, 40)   # 150 samples, 40 MFCC features
    y_audio = np.random.randint(0, 3, size=150)  # 3 audio classes

    audio_svm_metrics = ua.audio_svm_on_mfcc(X_audio, y_audio)
    print_metrics("AUDIO - SVM on MFCC", audio_svm_metrics)

    # ---------------- IMAGE-LIKE TEST DATA ----------------
    # Simulate CNN-embedding-like image features
    X_image = np.random.rand(120, 256)  # 120 samples, 256-dim features
    y_image = np.random.randint(0, 4, size=120)  # 4 image classes

    image_rf_metrics = ua.image_random_forest_on_features(X_image, y_image)
    print_metrics("IMAGE - Random Forest on features", image_rf_metrics)

    # ---------------- LOG-LIKE TEST DATA ----------------
    # Simulate TF-IDF-like log text features
    X_log = np.random.rand(180, 80)   # 180 samples, 80 features
    y_log = np.random.randint(0, 2, size=180)  # binary log classes

    log_lr_metrics = ua.log_text_logistic_regression(X_log, y_log)
    print_metrics("LOG - Logistic Regression", log_lr_metrics)

    # ---------------- SENSOR-LIKE TEST DATA ----------------
    # Simulate aggregated time-series features
    X_sensor = np.random.rand(160, 30)   # 160 samples, 30 features
    y_sensor = np.random.randint(0, 2, size=160)  # binary sensor states

    sensor_rf_metrics = ua.sensor_rf_on_features(X_sensor, y_sensor)
    print_metrics("SENSOR - Random Forest", sensor_rf_metrics)

    # ---------------- ANOMALY (UNSUPERVISED) EXAMPLE ----------------
    # IsolationForest (no accuracy metrics, just shows it runs)
    X_sensor_unlabeled = np.random.rand(100, 30)
    iso_result = ua.sensor_isolation_forest(X_sensor_unlabeled)
    print("=" * 60)
    print("SENSOR - Isolation Forest (unsupervised)")
    print(f"Scores shape: {iso_result['scores'].shape}")
    print()


if __name__ == "__main__":
    main()
