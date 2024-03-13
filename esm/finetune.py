from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from utils.fasta import load_gzdata
import torch
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

_, train_seqs = load_gzdata('/home/ubuntu/02712/data/ll_train.fa.gz', one_hot=False)
_, val_seqs = load_gzdata('/home/ubuntu/02712/data/ll_val.fa.gz', one_hot=False)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
train_tokenized = tokenizer(train_seqs)
val_tokenized = tokenizer(val_seqs)

# Create dataset for fine-tuning
train_dataset = Dataset.from_dict(train_tokenized)
val_dataset = Dataset.from_dict(val_tokenized)
del train_tokenized, val_tokenized, train_seqs, val_seqs

# Load model
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t30_150M_UR50D")

gc.collect()
torch.cuda.empty_cache()

train_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    remove_unused_columns=True,
)
trainer = Trainer(
    model,
    train_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("-----Start fine-tuning-----")
trainer.train()