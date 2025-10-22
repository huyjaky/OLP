from tokenizers import CharBPETokenizer

tokenizer = CharBPETokenizer()
tokenizer.train(
    files="../Attachments/NER_tokenizer/POS_tokens.txt",
    vocab_size=31000,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "[PAD]",
        "</s>",
        "[UNK]",
        "[MASK]",
    ],
)

tokenizer.save_model("../Attachments/NER_tokenizer")

