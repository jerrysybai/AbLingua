import argparse
import os
import warnings

from transformers import BertForMaskedLM

from .collate import Simple_Collator
from .tokenizer import BioTokenizer


def get_collator():
    parser = argparse.ArgumentParser()
    parser = Simple_Collator.add_args(parser)
    args = parser.parse_args([])
    tokenizer = BioTokenizer()
    collator = Simple_Collator(tokenizer, args)
    return collator


def get_model(weight_path: str = "AbLingua/weight"):
    if len(os.listdir(weight_path)) == 0:
        # json_url = 'https://huggingface.co/IDEA-XtalPi/AbLingua/resolve/main/config.json'
        # weight_url = 'https://huggingface.co/IDEA-XtalPi/AbLingua/resolve/main/pytorch_model.bin'

        warnings.warn("Weight files can not find!")
        return
    
    try:
        model = BertForMaskedLM.from_pretrained(
            weight_path, output_attentions=False, output_hidden_states=True,
        )
        return model
    except Exception as e:
        print(e/n)
        print("Model loading failed!")
