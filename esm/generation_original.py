import torch
import numpy as np
from torch.distributions.categorical import Categorical
from utils.fasta import output_fasta

CKPT_PATH = "test_trainer/checkpoint-103515"
SEQ_LEN = 360
SEQ_NUM = 1000

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t30_150M_UR50D")
model.to("cuda:0")
model.eval()
batch_converter = alphabet.get_batch_converter()

masked_seq = '<mask>'*(SEQ_LEN)
masked_data = [("protein 1", masked_seq)]
_, _, batch_tokens = batch_converter(masked_data)
batch_tokens = batch_tokens.cuda()

# Get logits
with torch.no_grad():
    results = model(batch_tokens)
logits = results["logits"]
probs = torch.nn.functional.softmax(logits, dim=-2)

idx_to_tok = {v: k for k, v in alphabet.tok_to_idx.items()}

# Gibbs Sampling
res_seqs = []
for _ in range(SEQ_NUM):
    sample = Categorical(probs.squeeze()).sample().cpu().detach().numpy()
    res_tmp = np.vectorize(idx_to_tok.get)(sample)[1:-1]
    res = ''.join(['-' if item in ('<cls>', '<pad>', '<eos>', '<mask>', '<unk>', '<null_1>', '.', 'X', 'B', 'U', 'Z', 'O') else item for item in res_tmp])
    res_seqs.append(res)

names = ['s{}'.format(i+1) for i in range(SEQ_NUM)]
output_fasta(names, res_seqs, "esm_original.fa")
