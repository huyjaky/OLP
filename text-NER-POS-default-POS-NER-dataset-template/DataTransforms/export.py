import pandas as pd

encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]

for enc in encodings:
    try:
        df = pd.read_csv("../Attachments/NER_dataset_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")

df['Sentence'].to_clipboard(index=False, header=False)
