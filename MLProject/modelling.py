import os
import argparse
from pathlib import Path

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def resolve_dataset_path(p: str) -> str:
    project_dir = Path(__file__).resolve().parent  # .../MLProject
    raw = Path(p)

    candidates = [
        raw,
        project_dir / raw,
        project_dir / str(p).replace("MLProject/", ""),
        project_dir.parent / raw,
        project_dir.parent / str(p).replace("MLProject/", ""),
    ]

    for c in candidates:
        if c.is_file():
            return str(c.resolve())

    raise FileNotFoundError(
        "Dataset tidak ditemukan.\n"
        f"Input: {p}\n"
        "Dicoba:\n- " + "\n- ".join(str(x) for x in candidates)
    )


def pick_text_column(df: pd.DataFrame) -> str:
    if "clean_review" in df.columns:
        return "clean_review"
    if "text" in df.columns:
        return "text"
    for c in df.columns:
        if df[c].dtype == "object":
            return c
    raise ValueError("Tidak ada kolom teks (clean_review/text) di dataset.")


def pick_target_column(df: pd.DataFrame) -> str:
    if "sentiment_label" in df.columns:
        return "sentiment_label"
    if "sentiment" in df.columns:
        return "sentiment"
    for c in ["target", "label", "class", "y", "output"]:
        if c in df.columns:
            return c
    return df.columns[-1]


def reshape_to_2d(x):
    """
    Fungsi TOP-LEVEL (bukan lambda) agar pipeline bisa dipickle.
    x akan berupa pandas Series / numpy array.
    """
    import numpy as np
    arr = np.asarray(x)
    return arr.reshape(-1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.getenv("DATASET_PATH", "youtube_comment_preprocessing/youtube_comment_preprocessed.csv"),
        help="Path dataset CSV. Default: youtube_comment_preprocessing/youtube_comment_preprocessed.csv",
    )
    parser.add_argument("--test_size", type=float, default=float(os.getenv("TEST_SIZE", "0.2")))
    parser.add_argument("--random_state", type=int, default=int(os.getenv("RANDOM_STATE", "42")))
    args = parser.parse_args()

    print("=" * 60)
    print("MLflow Training Pipeline Started (YouTube Comment Dataset)")
    print("=" * 60)

    mlflow.set_experiment("youtube_comment_classification")

    with mlflow.start_run():
        # [1] Load dataset
        print("\n[1/7] Loading dataset...")
        dataset_path = resolve_dataset_path(args.data_path)
        df = pd.read_csv(dataset_path)

        print(f"✓ Dataset loaded from: {dataset_path}")
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())

        # [2] Identify columns
        print("\n[2/7] Identifying target & feature columns...")
        target_col = pick_target_column(df)
        text_col = pick_text_column(df)
        use_text_length = "text_length" in df.columns

        print(f"✓ Target column : {target_col}")
        print(f"✓ Text column   : {text_col}")
        print(f"✓ Use length    : {use_text_length}")

        # [3] Prepare data
        print("\n[3/7] Preparing data...")
        y = df[target_col]

        feature_cols = [text_col] + (["text_length"] if use_text_length else [])
        X = df[feature_cols].copy()

        X[text_col] = X[text_col].fillna("").astype(str)
        if use_text_length:
            X["text_length"] = pd.to_numeric(X["text_length"], errors="coerce").fillna(0)

        print(f"✓ Features used: {feature_cols}")
        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y if len(pd.Series(y).unique()) > 1 else None,
        )

        print(f"✓ Train set: {X_train.shape}")
        print(f"✓ Test set : {X_test.shape}")

        # [4] Build pipeline
        print("\n[4/7] Building model pipeline...")

        transformers = [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), text_col),
        ]

        if use_text_length:
            transformers.append(
                ("len", FunctionTransformer(reshape_to_2d, validate=False), "text_length")
            )

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        clf = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", LogisticRegression(max_iter=2000, n_jobs=-1)),
            ]
        )

        # [5] Train
        print("\n[5/7] Training model...")
        clf.fit(X_train, y_train)
        print("✓ Model trained successfully")

        # [6] Evaluate
        print("\n[6/7] Evaluating model...")
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        print(f"✓ Training Accuracy: {train_acc:.4f}")
        print(f"✓ Testing Accuracy : {test_acc:.4f}")
        print(f"✓ Accuracy         : {acc:.4f}")

        # MLflow logging
        mlflow.log_param("dataset_path_resolved", dataset_path)
        mlflow.log_param("text_col", text_col)
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("use_text_length", use_text_length)
        mlflow.log_param("model_type", "LogisticRegression+TFIDF")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "1-2")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("accuracy", acc)

        # Save model
        print("\n[7/7] Saving model...")
        os.makedirs("models", exist_ok=True)
        model_path = "models/youtube_comment_model.pkl"
        joblib.dump(clf, model_path)
        print(f"✓ Model saved to: {model_path}")

        # Log model
        mlflow.sklearn.log_model(clf, "model")
        print("✓ Model logged to MLflow")

        print("\n" + "=" * 60)
        print("✓✓✓ Training Pipeline Completed Successfully! ✓✓✓")
        print("=" * 60)

        return acc


if __name__ == "__main__":
    final_acc = main()
    print(f"\nFinal Model Accuracy: {final_acc:.4f}")
