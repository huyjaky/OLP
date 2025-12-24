from transformers import BertConfig, BertForSequenceClassification, T5Tokenizer
# from get_test_train_dataset import id2sentiments

# tokenizer = T5Tokenizer.from_pretrained('./Attachments/spm_bpe_tokenizer.model')
# tokenizer.add_special_tokens({
#     "bos_token": "<s>",
#     "eos_token": "</s>",
#     "pad_token": "<pad>",
#     "unk_token": "<unk>"
# })
# model = BertForSequenceClassification.from_pretrained('./sentiment-bert-model_acc-0.8615967035293579_loss-0.5085833072662354/', num_labels=3)
#
# text = "This is a nerd guy"
# inputs = tokenizer("<s>"+text+"</s>", return_tensors="pt", padding=True, truncation=True, max_length=64)
# outputs = model(**inputs)
# logits = outputs.logits

from dotenv import load_dotenv
import os 
load_dotenv('../.env')

A = os.getenv('RAW_VAL_DATSET_PATH')
print(A)

