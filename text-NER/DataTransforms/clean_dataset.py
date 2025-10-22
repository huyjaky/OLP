import pandas as pd

encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]

for enc in encodings:
    try:
        df = pd.read_csv("../Attachments/NER_dataset.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")

df = df[~df["POS"].isin(["$", "|", "``", ".", ",", ":", ";"])]

# # Tạo mapping
id2tag = {k: v for k, v in enumerate(df["Tag"].unique())}
id2pos = {k: v for k, v in enumerate(df["POS"].unique())}
tag2id = {v: k for k, v in id2tag.items()}
pos2id = {v: k for k, v in id2pos.items()}
print("Tag2ID:", len(tag2id))
print("POS2ID:", len(pos2id))

#
# df = df.ffill()
#
# sentences = []
# tags_all = []
# poses_all = []
# tag_ids_all = []
# pos_ids_all = []
#
# grouped = df.groupby("Sentence #")
#
# for _, group in grouped:
#     words = " ".join(list(group["Word"]))
#     tag_list = list(group["Tag"])
#     pos_list = list(group["POS"])
#
#     tag_id_list = [tag2id[t] for t in tag_list]
#     pos_id_list = [pos2id[p] for p in pos_list]
#
#     sentences.append(words)
#     tags_all.append(tag_list)
#     poses_all.append(pos_list)
#     tag_ids_all.append(tag_id_list)
#     pos_ids_all.append(pos_id_list)
#
# df_new = pd.DataFrame({
#     "Sentence": sentences,
#     "Tags": tags_all,
#     "POS": poses_all,
#     "Tag_id": tag_ids_all,
#     "POS_id": pos_ids_all
# })
#
# # Xuất ra file
# df_new.to_csv("../Attachments/NER_dataset_cleaned.csv", index=False)
