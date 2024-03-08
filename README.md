# BiMamba

This repository wraps a bidirectional Mamba module in Hugging Face compatible APIs / classes.

To use BiMamba as a drop-in replacement for other Hugging Face models, you can use the following code:

```python
"""Sample code for initializing BiMamba from the template HF hub model."""

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model_name_or_path = "yairschiff/bimamba-template"
config_overrides = {
    "d_model": 128,  # TODO: Change this as desired
    "n_layer": 2,  # TODO: Change this as desired
    "pad_token_id": tokenizer.pad_token_id,
    "vocab_size": tokenizer.vocab_size,
    "pad_vocab_size_multiple": 1,
    # TODO: See configuration_bimamba for all config options
}
config = AutoConfig.from_pretrained(
    model_name_or_path,
    **config_overrides,
    trust_remote_code=True
)
model = AutoModelForMaskedLM.from_config(
    config=config,
    trust_remote_code=True
)

# Test the model
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = ["A sample sentence for model testing."]
tokenized = tokenizer.batch_encode_plus(inputs, return_tensors="pt")
model = model.to(device)
model_out = model(tokenized["input_ids"].to(device))
```