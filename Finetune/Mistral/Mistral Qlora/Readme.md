## Finetuning With Peft Qlora

### Introduction

we fine-tune the model on new data, this fine-tuning takes much less data and computation compared to the previous steps but it’s still not possible on consumer-grade hardware.

PEFT(parameter efficient fine-tuning) fixes this issue and applies clever methods to make fine-tuning possible even on the free tier of Google Colab.

We’ll use Quantization and LoRA(Low-Rank Adaptation) to fine-tune Mistral 7b instruct and introduce new knowledge to it.

### Standard Mistral Dataset Format

Mistral 7b instruct uses a much simpler format. At first, we have the BOS(begin of sequence) token which is `<s>`, there is no system message, and then the user's prompt goes between `[INST]` and `[/INST]`, then there is the assistant’s response and at the end, we have the EOS(end of sequence) token `</s>`.

This is an example of a generated text in Mistral’s format.

```
<s>[INST] What is your favourite condiment? [/INST]
Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s>
```

#### Requirements

    ```
    !git clone 'https://github.com/ali7919/Enlighten-Instruct.git'
    !pip install -U bitsandbytes
    !pip install transformers==4.36.2
    !pip install -U peft
    !pip install -U accelerate
    !pip install -U trl
    !pip install datasets==2.16.0
    !pip install sentencepiece

```
### Dataset

We use a processed version of the Dataset for our Mistral model
You can look up the Github Repo - https://github.com/ali7919/Enlighten-Instruct/blob/main/DataGenerator.ipynb. to check how the data is being converted to standard format for Mistal.

Dataset Link:
https://github.com/ali7919/Enlighten-Instruct.git
```
