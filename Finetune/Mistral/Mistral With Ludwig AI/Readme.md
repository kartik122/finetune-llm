## Finetuning With Ludwig AI

### Ludwig

Ludwig is a low-code framework for building custom AI models like LLMs and other deep neural networks.

Key features:

🛠 Build custom models with ease: a declarative YAML configuration file is all you need to train a state-of-the-art LLM on your data. Support for multi-task and multi-modality learning. Comprehensive config validation detects invalid parameter combinations and prevents runtime failures.
⚡ Optimized for scale and efficiency: automatic batch size selection, distributed training (DDP, DeepSpeed), parameter efficient fine-tuning (PEFT), 4-bit quantization (QLoRA), paged and 8-bit optimizers, and larger-than-memory datasets.
📐 Expert level control: retain full control of your models down to the activation functions. Support for hyperparameter optimization, explainability, and rich metric visualizations.
🧱 Modular and extensible: experiment with different model architectures, tasks, features, and modalities with just a few parameter changes in the config. Think building blocks for deep learning.
🚢 Engineered for production: prebuilt Docker containers, native support for running with Ray on Kubernetes, export models to Torchscript and Triton, upload to HuggingFace with one command.

### Training with Ludwig

To train a model with Ludwig, we first need to create a Ludwig configuration. The config specifies input features, output features, preprocessing, model architecture, training loop, hyperparameter search, and backend infrastructure -- everything that's needed to build, train, and evaluate a model.

Sample Yaml:

```
input_features:
    - name: genres
      type: set
      preprocessing:
          tokenizer: comma
    - name: content_rating
      type: category
    - name: top_critic
      type: binary
    - name: runtime
      type: number
    - name: review_content
      type: text
      encoder:
          type: embed
output_features:
    - name: recommended
      type: binary
```

You can tweak the Yaml inside the code for increasing or decreasing the model training parameters.

### Getting Started

#### Requirements

    - pip install ludwig ludwig[llm] peft
    - Small GPU/ Colab Notebook for 7B models, for 13B and above can run on Nvidia L4 or higher depending upon the config

### Dataset

We use the Alpaca Instruction Following Dataset which is used to provide instruction following LLM.
https://huggingface.co/datasets/tatsu-lab/alpaca
