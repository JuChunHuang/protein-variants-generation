import os
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from transformers import AutoModelForMaskedLM, AutoTokenizer
from utils.preprocess import tok_to_idx
from utils.fasta import output_fasta

CKPT_PATH = "test_trainer/checkpoint-103515"
SEQ_LEN = 360
SEQ_NUM = 1000

tok_to_idx = tok_to_idx(os.path.join(CKPT_PATH, "vocab.txt"))

seq = 'X'*SEQ_LEN

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH)
model = AutoModelForMaskedLM.from_pretrained(CKPT_PATH)

# Tokenization
input_ids = tokenizer.encode(seq, return_tensors="pt")
sequence_length = input_ids.shape[1] - 2
position = torch.tensor(range(10, sequence_length+1))
masked_input_ids = input_ids.clone()
masked_input_ids[0, position] = tokenizer.mask_token_id
model.eval()

# Get logits
with torch.no_grad():
    logits = model(masked_input_ids).logits

probs = torch.nn.functional.softmax(logits, dim=-2)

# Gibbs Sampling
res_seqs = []
for _ in range(SEQ_NUM):
    sample = Categorical(probs.squeeze()).sample().cpu().detach().numpy()
    res_tmp = np.vectorize(tok_to_idx.get)(sample)[1:-1]
    res = ''.join(['-' if item in ('<cls>', '<pad>', '<eos>', '<mask>', '<unk>', '<null_1>', '.', 'X', 'B', 'U', 'Z', 'O') else item for item in res_tmp])
    res_seqs.append(res)

names = ['s{}'.format(i+1) for i in range(SEQ_NUM)]
output_fasta(names, res_seqs, "esm_finetune.fa")
