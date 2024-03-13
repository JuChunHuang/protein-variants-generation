import numpy as np

def tok_to_idx(file_path):
    token_dict = {}
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            token = line.strip()
            token_dict[index] = token

    return token_dict

def seq_to_one_hot(sequence, aa_key):
    arr = np.zeros((len(sequence),len(aa_key)))
    for j, c in enumerate(sequence):
        arr[j, aa_key[c]] = 1
    return arr

def to_one_hot(seqlist, alphabet=['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                                  'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']):
    aa_key = {l: i for i, l in enumerate(alphabet)}
    if type(seqlist) == str:
        return seq_to_one_hot(seqlist, aa_key)
    else:
        encoded_seqs = []
        for prot in seqlist:
            encoded_seqs.append(seq_to_one_hot(prot, aa_key))
        return np.stack(encoded_seqs)