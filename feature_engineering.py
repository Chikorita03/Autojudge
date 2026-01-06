import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

def build_features(df):

    df["text_length"] = df["combined_text"].str.len()
    df["word_count"] = df["combined_text"].str.split().str.len()
    df["avg_word_length"] = df["text_length"] / (df["word_count"] + 1)

    df["comparison_symbols"] = df["combined_text"].str.count(r"[<>=]")
    df["arithmetic_symbols"] = df["combined_text"].str.count(r"[+\-*/%]")

    keywords = [
        "graph", "tree", "dfs", "bfs",
        "dp", "dynamic programming",
        "binary search", "greedy",
        "recursion", "bitmask"
    ]

    for kw in keywords:
        safe_kw = kw.replace(" ", "_")
        pattern = r"\b" + re.escape(kw) + r"\b"

        df[f"freq_{safe_kw}"] = df["combined_text"].str.count(pattern)
        df[f"has_{safe_kw}"] = (df[f"freq_{safe_kw}"] > 0).astype(int)

    numeric_columns = (
        ["text_length", "word_count", "avg_word_length",
         "comparison_symbols", "arithmetic_symbols"] +
        [f"freq_{kw.replace(' ', '_')}" for kw in keywords] +
        [f"has_{kw.replace(' ', '_')}" for kw in keywords]
    )

    numeric_features = df[numeric_columns].values

    X_text = df["combined_text"]

    X_text_train, X_text_test, idx_train, idx_test = train_test_split(
        X_text,
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=df["problem_class"]
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_tfidf = tfidf.fit_transform(X_text_train)
    X_test_tfidf = tfidf.transform(X_text_test)

    num_train = numeric_features[idx_train]
    num_test = numeric_features[idx_test]

    scaler = StandardScaler()
    num_train_scaled = scaler.fit_transform(num_train)
    num_test_scaled = scaler.transform(num_test)

    X_train = hstack([X_train_tfidf, num_train_scaled])
    X_test = hstack([X_test_tfidf, num_test_scaled])

    return X_train, X_test, idx_train, idx_test, tfidf, scaler

def build_features_for_inference(df, tfidf, scaler):

    X_text = tfidf.transform(df["combined_text"])

    df["text_length"] = df["combined_text"].str.len()
    df["word_count"] = df["combined_text"].str.split().str.len()
    df["avg_word_length"] = df["text_length"] / (df["word_count"] + 1)

    df["comparison_symbols"] = df["combined_text"].str.count(r"[<>=]")
    df["arithmetic_symbols"] = df["combined_text"].str.count(r"[+\-*/%]")

    keywords = [
        "graph", "tree", "dfs", "bfs",
        "dp", "dynamic programming",
        "binary search", "greedy",
        "recursion", "bitmask"
    ]

    for kw in keywords:
        safe_kw = kw.replace(" ", "_")
        df[f"freq_{safe_kw}"] = df["combined_text"].str.count(r"\b" + kw + r"\b")
        df[f"has_{safe_kw}"] = (df[f"freq_{safe_kw}"] > 0).astype(int)

    numeric_columns = (
        ["text_length", "word_count", "avg_word_length",
         "comparison_symbols", "arithmetic_symbols"] +
        [f"freq_{kw.replace(' ', '_')}" for kw in keywords] +
        [f"has_{kw.replace(' ', '_')}" for kw in keywords]
    )

    X_num = scaler.transform(df[numeric_columns].values)

    return X_text, X_num