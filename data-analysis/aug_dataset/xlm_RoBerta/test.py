from transformers import XLMRobertaTokenizer


tokenizer = XLMRobertaTokenizer.from_pretrained("./tokenizer_model/spm_bpe_tokenizer.model")
tokenizer.add_special_tokens({
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>"
})

# Test the tokenizer
sample_text = "This is a sample text for tokenization."
encoded = tokenizer.encode(sample_text)
decoded = tokenizer.decode(encoded)
print("Encoded:", encoded)
print("Decoded:", decoded)

