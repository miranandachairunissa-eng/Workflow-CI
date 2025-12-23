import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "youtube_comment_preprocessing/youtube_comment_preprocessed.csv"

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("youtube_sentiment_baseline")
    mlflow.sklearn.autolog() # Syarat Basic

    data = pd.read_csv(DATA_PATH)
    X = data["text"].astype(str)
    y = data["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="baseline_run"):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("logreg", LogisticRegression(max_iter=1000))
        ])
        print("Menjalankan Baseline Model...")
        pipeline.fit(X_train, y_train)

if __name__ == "__main__":
    main()
