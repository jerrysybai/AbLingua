# AbLinuga: Antibody Language Model with Linguistic Tokenization

## Main models you should use
| Model | Dataset | Description|
|-------|---------|------------|
|AbLingua-300M|OAS-Unpaired-300M|A small language model that is easy to fine-tune|
|AbLingua-600M|OAS-Unpaired-300M|Medium-sized model that can support a variety of downstream tasks and performs well in antibody structure prediction|

## Install
As a prerequisite, you must have `PyTorch` and `Transformers` installed to use this repository. `PyTorch` and `Transformers` only require the latest version to be installed.

```bash
# Python >= 3.10
pip install torch
pip install transformers
```

## Download language model weight
```bash
# weight should in weight floder
cd AbLingua/weight/
# download model json
wget -c 'https://huggingface.co/IDEA-XtalPi/AbLingua/resolve/main/config.json'
# download model weight
wget -c 'https://huggingface.co/IDEA-XtalPi/AbLingua/resolve/main/pytorch_model.bin'
```

## Usage




### Antibody sequence embedding
Representations from AbLinuga may be useful as features for deep learning models.
```python
from AbLingua.embed import get_collator, get_model

collator = get_collator()
model = get_model()

seq = ['QVTLRESGPAL', 
       'VKPTQTLTLTC']
seq_input = collator(seq)

tokens_embedding = model(**seq_input).hidden_states[-1]

# tokens_embedding.shape
# [2, 256, 1280]
```

## Citation
