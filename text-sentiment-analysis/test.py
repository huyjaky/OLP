import pandas as pd

encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]

for enc in encodings:
    try:
        df_train = pd.read_csv("../Attachments/train_cleaned.csv", encoding=enc).dropna()
        # df = pd.read_csv("../Attachments/NER_dataset_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df_train.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")

print(df_train[["text", "sentiment", "Time of Tweet"]])

