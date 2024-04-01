## Finetuning Microsoft's Phi - 2

### Phi - 2

With 2.7 billion parameters, Phi 2 proves to be extremely small and useful as it would fit within the capabilities of a Google Colab workspace, which offers a Tesla T4 GPU, 16 GB of RAM, and ample disk space for training and storing the model.

### Training with Qlora

Quantization is a method that enables us to represent information using fewer bits, although at the expense of reduced precision. For instance, when storing model weights in 32-bit floating points, each weight occupies 4 bytes of memory. By employing quantization, we can opt for a 16-bit representation, halving the memory requirement to 2 bytes, or an 8-bit representation, quartering it to 1 byte. For a more aggressive reduction, we can further opt for a 4-bit representation, utilizing only 0.5 bytes. This process is valuable for optimizing memory usage and computational efficiency, particularly when deploying models on resource-constrained devices or aiming for faster inference speeds.

LoRA is an exceptionally efficient technique for fine-tuning a pretrained language model without requiring updates to all parameters of the entire model. The key principle involves updating only a small batch of low-rank matrices that are appended to the existing weights. While fine-tuning with LoRA may exhibit slightly lower performance compared to full fine-tuning, the advantages in terms of performance retention and training speed far outweigh any drawbacks.

### Getting Started

#### Requirements

    ```
    !pip install einops datasets bitsandbytes accelerate peft flash_attn
    !pip uninstall -y transformers
    !pip install git+https://github.com/huggingface/transformers
    !pip install --upgrade torch
    ```
    - Small GPU/ Colab Notebook for 7B models, for 13B and above can run on Nvidia L4 or higher depending upon the config
    Note: On a T4 GPU, if the max length exceed 1024, it will throw an Out-Of-Memory exception. The model will train very slow on a T4 GPU.

### Dataset

We use the WebGlm-QA Dataset (https://huggingface.co/datasets/THUDM/webglm-qa),
Consists of thousands of high-quality data samples in a question-answer format, complete with references to generate the answers. The references are quoted in the answer.
We are taking only a slice of data from training dataset, about 5000 rows and we are merging validate and test datasets, which amount to 1400 rows.
