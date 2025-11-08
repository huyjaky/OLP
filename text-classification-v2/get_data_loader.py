import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from get_tokenizer import create_tokenizer
from norm_text import text_normalize
import torch
from tokenizers import Tokenizer


encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]


for enc in encodings:
    try:
        df = pd.read_csv("./Attachments/train.csv", encoding=enc).dropna()
        # df = pd.read_csv("../Attachments/NER_dataset_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")


for enc in encodings:
    try:
        df_test = pd.read_csv("./Attachments/test.csv", encoding=enc).dropna()
        # df = pd.read_csv("../Attachments/NER_dataset_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df_test.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")

df_0 = df[df["target"] == 0].iloc[:80000]  # type: ignore
df_1 = df[df["target"] == 1]  # type: ignore
df = pd.concat([df_0, df_1])
df = df.sample(frac=1).reset_index(drop=True)


class CustomCollator(Dataset):
    def __init__(self, questions, target, tokenizer, is_training=True):
        self.questions = questions
        self.target = target
        self.tokenizer = tokenizer
        self.is_training = is_training

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):

        question = text_normalize(self.questions[idx])
        encoding = self.tokenizer.encode(question).ids

        if self.is_training:
            target_id = [0, 0]
            target_id[int(self.target[idx])] = 1
            return (
                torch.tensor(encoding, dtype=torch.long),
                torch.tensor(target_id, dtype=torch.float),
            )
        else:
            return torch.tensor(encoding, dtype=torch.long)


tokenizer = create_tokenizer()
# tokenizer = Tokenizer.from_file("./Attachments/tokenizer-wp.json")

X_train, X_val, y_train, y_val = train_test_split(
    df["question_text"].apply(lambda x: str(x)).tolist(),  # type: ignore
    df["target"].apply(lambda x: int(x)).tolist(),  # type: ignore
    test_size=0.2,
    random_state=42,
)

X_test = df_test["question_text"].apply(lambda x: str(x)).tolist()  # type: ignore

training_dataset = CustomCollator(X_train, y_train, tokenizer)  # type: ignore
validation_dataset = CustomCollator(X_val, y_val, tokenizer)  # type: ignore
testing_dataset = CustomCollator(X_test, y_val, tokenizer, is_training=False)  # type: ignore

training_loader = DataLoader(
    training_dataset, batch_size=128, shuffle=True, drop_last=True
)  # type: ignore

validation_loader = DataLoader(
    validation_dataset, batch_size=128, shuffle=False, drop_last=True
)  # type: ignore

testing_loader = DataLoader(
    testing_dataset, batch_size=128, shuffle=False, drop_last=False
)  # type: ignore

print("huy")
