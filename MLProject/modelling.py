import os
import argparse
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


def _pick_text_column(df: pd.DataFrame) -> str:
    """Pilih kolom teks terbaik."""
    if "clean_review" in df.columns:
        return "clean_review"
    if "text" in df.columns:
        return "text"
    # fallback: cari kolom object/string pertama
    for c in df.columns:
        if df[c].dtype == "object":
            return c
    raise ValueError("Tidak menemukan kolom teks (clean_review/text) di dataset.")


def _pick_target_column(df: pd.DataFrame) -> str:
    """Pilih kolom target terbaik."""
    if "sentiment_label" in df.columns:
        return "sentiment_label"
    if "sentiment" in df.columns:
        return "sentiment"
    # fallback: cek beberapa kemungkinan
    possible_targets = ["target", "label", "class", "y", "output"]
    for c in possible_targets:
        if c in df.columns:
            return c
    # terakhir: pakai kolom terakhir
    return df.columns[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.getenv(
            "DATASET_PATH",
            "youtube_comment_preprocessing/youtube_comment_preprocessed.csv"
        ),
        help="Path dataset CSV (default: youtube_comment_preprocessing/youtube_comment_preprocessed.csv)",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("MLflow Training Pipeline Started (YouTube Comment Dataset)")
    print("=" * 60)

    # ✅ Ganti nama experiment supaya konsisten dengan project kamu
    mlflow.set_experiment("youtube_comment_classification")

    with mlflow.start_run():
        # Load dataset
        print("\n[1/6] Loading dataset...")
        df = pd.read_csv(args.data_path)
        print(f"✓ Dataset loaded from: {args.data_path}")
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())

        # Identify target & text
        print("\n[2/6] Identifying target & feature columns...")
        target_col = _pick_target_column(df)
        text_col = _pick_text_column(df)

        print(f"✓ Target column: {target_col}")
        print(f"✓ Text column: {text_col}")

        # Prepare X, y
        print("\n[3/6] Preparing data...")
        y = df[target_col]

        # fitur yang dipakai:
        # - text_col (clean_review/text) -> TF-IDF
        # - text_length (kalau ada) -> numeric
        use_text_length = "text_length" in df.columns

        feature_cols = [text_col] + (["text_length"] if use_text_length else [])
        X = df[feature_cols].copy()

        # Pastikan text kolom string, isi NaN jadi ""
        X[text_col] = X[text_col].fillna("").astype(str)

        if use_text_length:
            # Pastikan numeric
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
            stratify=y if len(pd.Series(y).unique()) > 1 else None
        )
        print(f"✓ Train set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")

        # Build pipeline
        print("\n[4/6] Building model pipeline...")
        transformers = []

        # TF-IDF untuk teks
        transformers.append(
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), text_col)
        )

        # Numeric (opsional): text_length
        if use_text_length:
            # ColumnTransformer butuh array 2D untuk numeric, biar aman:
            transformers.append(
                ("num", FunctionTransformer(lambda x: x.values.reshape(-1, 1), validate=False), "text_length")
            )

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        model = LogisticRegression(
            max_iter=2000,
            n_jobs=-1
        )

        clf = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model)
        ])

        # Train
        print("\n[5/6] Training model...")
        clf.fit(X_train, y_train)
        print("✓ Model trained successfully")

        # Evaluate
        print("\n[6/6] Evaluating model...")
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        print(f"✓ Training Accuracy: {train_acc:.4f}")
        print(f"✓ Testing Accuracy : {test_acc:.4f}")
        print(f"✓ Accuracy         : {accuracy:.4f}")

        # Log params & metrics
        mlflow.log_param("dataset_path", args.data_path)
        mlflow.log_param("text_col", text_col)
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("use_text_length", use_text_length)
        mlflow.log_param("model_type", "LogisticRegression + TFIDF")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "1-2")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("accuracy", accuracy)

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/youtube_comment_model.pkl"
        joblib.dump(clf, model_path)
        print(f"\n✓ Model saved to: {model_path}")

        # Log model to MLflow
        mlflow.sklearn.log_model(clf, "model")
        print("✓ Model logged to MLflow")

        print("\n" + "=" * 60)
        print("✓✓✓ Training Pipeline Completed Successfully! ✓✓✓")
        print("=" * 60)

        return accuracy


if __name__ == "__main__":
    acc = main()
    print(f"\nFinal Model Accuracy: {acc:.4f}")
