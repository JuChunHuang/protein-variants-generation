from transformers import pipeline
from utils.fasta import output_fasta

SEQ_NUM = 1000
SEQ_LENGTH = 360
protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")
sequences = protgpt2("<|endoftext|>", max_length=SEQ_LENGTH, min_length=SEQ_NUM, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=SEQ_NUM, eos_token_id=0)
res_seqs = []
for seq in sequences:
    output = seq["generated_text"][14:].replace("\n", "")
    res_seqs.append(output)

names = ['s{}'.format(i+1) for i in range(SEQ_NUM)]
output_fasta(names, res_seqs, "protgpt2_ori.fa")