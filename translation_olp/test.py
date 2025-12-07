from transformers import T5Tokenizer 

vi_token = T5Tokenizer(vocab_file="./Attachments/tgt_tokenizer.json")
print(vi_token.vocab_size)