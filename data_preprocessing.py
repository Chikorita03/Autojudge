import pandas as pd

def preprocess_data(path):
    df = pd.read_json(path, lines=True)

    df["combined_text"] = (
        df["title"] + " " +
        df["description"] + " " +
        df["input_description"] + " " +
        df["output_description"]
    )

    df["combined_text"] = (
        df["combined_text"]
        .str.lower()
        .str.replace("\n", " ", regex=False)
        .str.replace("\t", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    return df