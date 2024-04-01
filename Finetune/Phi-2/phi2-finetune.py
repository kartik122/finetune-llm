import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

model_name = "microsoft/phi-2"
# Configuration to load model in 4-bit quantized
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_compute_dtype='float16',
                                #bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True)


#Loading Microsoft's Phi-2 model with compatible settings
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto',
                                             quantization_config=bnb_config,
                                             attn_implementation="flash_attention_2",
                                             trust_remote_code=True)

# Setting up the tokenizer for Phi-2
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          add_eos_token=True,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

from datasets import load_dataset

#Load a slice of the WebGLM dataset for training and merge validation/test datasets
train_dataset = load_dataset("THUDM/webglm-qa", split="train[5000:10000]")
test_dataset = load_dataset("THUDM/webglm-qa", split="validation+test")

#Function that creates a prompt from instruction, context, category and response and tokenizes it
def collate_and_tokenize(examples):

    question = examples["question"][0].replace('"', r'\"')
    answer = examples["answer"][0].replace('"', r'\"')
    #unpacking the list of references and creating one string for reference
    references = '\n'.join([f"[{index + 1}] {string}" for index, string in enumerate(examples["references"][0])])

    #Merging into one prompt for tokenization and training
    prompt = f"""###System:
Read the references provided and answer the corresponding question.
###References:
{references}
###Question:
{question}
###Answer:
{answer}"""

    #Tokenize the prompt
    encoded = tokenizer(
        prompt,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        ## Very critical to keep max_length at 1024.
        ## Anything more will lead to OOM on T4
        max_length=2048,
    )

    encoded["labels"] = encoded["input_ids"]
    return encoded

#We will just keep the input_ids and labels that we add in function above.
columns_to_remove = ["question","answer", "references"]

#tokenize the training and test datasets
tokenized_dataset_train = train_dataset.map(collate_and_tokenize,
                                            batched=True,
                                            batch_size=1,
                                            remove_columns=columns_to_remove)
tokenized_dataset_test = test_dataset.map(collate_and_tokenize,
                                          batched=True,
                                          batch_size=1,
                                          remove_columns=columns_to_remove)

#Check if tokenization looks good
input_ids = tokenized_dataset_train[1]['input_ids']

decoded = tokenizer.decode(input_ids, skip_special_tokens=True)

#Accelerate training models on larger batch sizes, we can use a fully sharded data parallel model.
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

from peft import prepare_model_for_kbit_training

#gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Freeze base model layers and cast layernorm in fp32
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
print(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
    'q_proj',
    'k_proj',
    'v_proj',
    'dense',
    'fc1',
    'fc2',
    ], #print(model) will show the modules to use
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

lora_model = get_peft_model(model, config)
# print_trainable_parameters(lora_model)


lora_model = accelerator.prepare_model(lora_model)


import time
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',  # Output directory for checkpoints and predictions
    overwrite_output_dir=True, # Overwrite the content of the output directory
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    gradient_accumulation_steps=8, # number of steps before optimizing
    gradient_checkpointing=True,   # Enable gradient checkpointing
    gradient_checkpointing_kwargs={"use_reentrant": False},
    warmup_steps=50,  # Number of warmup steps
    max_steps=10,  # Total number of training steps
    num_train_epochs=1,  # Number of training epochs
    learning_rate=5e-5,  # Learning rate
    weight_decay=0.01,  # Weight decay
    optim="paged_adamw_8bit", #Keep the optimizer state and quantize it
    fp16=True, #Use mixed precision training
    #For logging and saving
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,  # Limit the total number of checkpoints
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True, # Load the best model at the end of training
)

trainer = Trainer(
    model=lora_model,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_test,
    args=training_args,
)

#Disable cache to prevent warning, renable for inference
#model.config.use_cache = False

start_time = time.time()  # Record the start time
trainer.train()  # Start training
end_time = time.time()  # Record the end time

training_time = end_time - start_time  # Calculate total training time

print(f"Training completed in {training_time} seconds.")

#Save model to hub to ensure we save our work.
# lora_model.push_to_hub("phi2-qlora-kg99",
#                   use_auth_token=True,
#                   commit_message="Training Phi-2",
#                   private=True)


