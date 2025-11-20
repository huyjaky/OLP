from model_backbone import MyModel
from tokenizers import Tokenizer


src_tokenizer = Tokenizer.from_file("./Attachments/tokenizer-level.json")

sentence = "Do đó, doanh nghiệp cần chú trọng đến yếu tố này để là thỏa mãn, hài lòng các nhóm đối tượng này."
model = MyModel.load_from_checkpoint(
    "./checkpoints/spelling_correction/spelling_correction-epoch=04-val_acc=0.96.ckpt",
    input_dim=64,
    output_dim=2,
    vocab_size=3000,
    tokenizer=src_tokenizer,
)

logits = model.predict_step(sentence)
print(logits)


