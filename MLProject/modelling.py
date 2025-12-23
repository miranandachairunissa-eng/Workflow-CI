import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Path disesuaikan untuk dijalankan di dalam folder MLProject
DATA_PATH = "youtube_comment_preprocessing/youtube_comment_preprocessed.csv"

def main():
    # Cek keberadaan file dan ukurannya untuk debugging
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File tidak ditemukan di {DATA_PATH}")
        return
    
    file_size = os.path.getsize(DATA_PATH)
    print(f"Membaca file: {DATA_PATH} ({file_size} bytes)")

    mlflow.set_tracking_uri("file:../mlruns") # Simpan di luar folder MLProject agar mudah di-upload
    mlflow.set_experiment("youtube_sentiment_baseline")
    mlflow.sklearn.autolog()

    try:
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
            print("Training Selesai!")
    except Exception as e:
        print(f"Terjadi kesalahan saat membaca data: {e}")

if __name__ == "__main__":
    main()
