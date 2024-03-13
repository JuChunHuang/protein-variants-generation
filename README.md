# Generating Protein Variants with Different Generative Models

Simple overview of use/purpose.


## Getting Started


### prerequisites

You should create a Python virtual environment. After you are ready, run

`pip install -r requirements.txt`

## Protein Sequences Generation

### ESM2
If you want to use the pretrained ESM2 model developed by Meta to generate de novo protein sequences, run

`esm/generation_original.py`

To finetune the pretrained ESM2 model on a set of user-defined protein sequences, run

`esm/finetune.py`

Then use your fine-tuned model to generate random protein sequences

`esm/generation_finetune.py`

### ProtGPT2
ProtGPT2 Accepts the dataset in completely like a fasta format, with a newline character every 60 aminoacids.

```
<|endoftext|>

MKDIDTLISNNALWSKMLVEEDPGFFEKLAQAQKPRFLWIGCSDSRVPAERLTGLEPGEL

FVHRNVANLVIHTDLNCLSVVQYAVDVLEVEHIIICGHYGCGGVQAAVENPELGLINNWL

LHIRDIWFKHSSLLGEMPQERRLDTLCELNVMEQVYNLGHSTIMQSAWKRGQKVTIHGWA

YGIHDGLLRDLDVTATNRETLEQRYRHGISNLKLKHANHK


<|endoftext|>

#ANOTHER SEQUENCE
```

Therefore, to reformat the dataset using the bacterial luciferase dataset, run

`protgpt2/preprocessing.py`

If you want to use the pretrained ProtGPT2 to generate <em>de novo<em> protein sequences, run

`protgpt2/generation_original.py`

To finetune the pretrained ProtGPT2 model on a set of user-defined protein sequences, run 

`python protgpt2/run_clm.py --model_name_or_path nferruz/ProtGPT2 --train_file train.txt --validation_file validation.txt --tokenizer_name nferruz/ProtGPT2
--do_train --do_eval --output_dir output --learning_rate 1e-06`

You can add ` --per_device_train_batch_size 4` if you got CUDA out of memory in PyTorch.


## Authors

Ju-Chun Huang, Lilin Huang


## Acknowledgments
* [ESM2](https://github.com/facebookresearch/esm)
* [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2)
* [Deep Protein Generation](https://github.com/alex-hh/deep-protein-generation)
