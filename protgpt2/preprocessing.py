from utils.fasta import load_gzdata, output_fasta

amino_acid = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
              'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def insert_newlines(seq, every=60):
    return '\n'.join(seq[i:i+every] for i in range(0, len(seq), every))

_, train_seqs = load_gzdata('/home/ubuntu/02712/data/ll_train.fa.gz', one_hot=False)
_, val_seqs = load_gzdata('/home/ubuntu/02712/data/ll_val.fa.gz', one_hot=False)

train_seqs = [insert_newlines(train_seqs[i]) for i in range(len(train_seqs))]
val_seqs = [insert_newlines(val_seqs[i]) for i in range(len(val_seqs))]

# Reformat training and validation data
train_names = ["<|endoftext|>" for _ in range(len(train_seqs))]
output_fasta(train_names[:len(train_names)], train_seqs[:len(train_seqs)], "train.txt")
val_names = ["<|endoftext|>" for _ in range(len(val_seqs))]
output_fasta(val_names[:len(val_names)], val_seqs[:len(val_seqs)], "validation.txt")