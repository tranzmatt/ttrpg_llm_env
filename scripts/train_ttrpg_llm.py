# TTRPG LLM Training Script
# Fine-tune a language model for Game Master assistance using Unsloth

import os
import json
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset, load_dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
import gc

# Install required packages (run these in terminal first):
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes


def setup_model_for_training(model_name: str = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit", 
                           max_seq_length: int = 2048):
    """
    Setup model optimized for RTX 3000 series (12-16GB VRAM)
    
    Args:
        model_name: Model to use for training
        max_seq_length: Maximum sequence length
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")
    
    # Model options optimized for 12-16GB VRAM
    model_options = {
        "mistral-7b": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "llama2-7b": "unsloth/llama-2-7b-chat-bnb-4bit", 
        "qwen2.5-7b": "unsloth/qwen2.5-7b-instruct-bnb-4bit",
        "phi3-mini": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"  # Smaller option
    }
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect (Float16 for Tesla T4, V100, Bfloat16 for Ampere+)
        load_in_4bit=True,  # Essential for consumer GPUs
        # token="hf_...", # Use if using gated models like meta-llama/Llama-2-7b-hf
    )
    
    # Setup LoRA for parameter-efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank - higher = more parameters but better quality
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,  # LoRA scaling factor
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",  # Bias type
        use_gradient_checkpointing="unsloth",  # Memory optimization
        random_state=3407,
        use_rslora=False,  # Support rank stabilized LoRA
        loftq_config=None,  # LoftQ quantization
    )
    
    return model, tokenizer

def prepare_dataset(data_file: str, tokenizer, max_samples: int = None):
    """
    Prepare the dataset for training
    
    Args:
        data_file: Path to JSON training data
        tokenizer: Model tokenizer
        max_samples: Maximum number of samples to use (for testing)
        
    Returns:
        Prepared dataset
    """
    print(f"Loading dataset from: {data_file}")
    
    # Load the JSON data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Using {max_samples} samples for training")
    
    print(f"Total training examples: {len(data)}")
    
    # Convert to Hugging Face dataset
    dataset = Dataset.from_list(data)
    
    # Add chat template to tokenizer
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",  # or "llama-2", "chatml", etc.
        mapping={
            "system": "role",
            "user": "role",
            "assistant": "role",
            "role" : "role",
            "content": "content"
        },
    )
    
    def formatting_prompts_func(examples):
        """Format the data for training"""
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}
    
    # Apply formatting
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    return dataset

def train_ttrpg_model(model, tokenizer, dataset, output_dir: str = "./ttrpg_gm_model"):
    """
    Train the TTRPG GM model with memory optimization for RTX 3000 series
    
    Args:
        model: The model to train
        tokenizer: Model tokenizer  
        dataset: Training dataset
        output_dir: Directory to save the trained model
        
    Returns:
        Trained model
    """
    print("Starting training...")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments optimized for 12-16GB VRAM
    training_args = TrainingArguments(
        per_device_train_batch_size=1,  # Critical for memory management
        gradient_accumulation_steps=4,  # Effective batch size = 4
        warmup_steps=100,
        num_train_epochs=3,  # Adjust based on dataset size
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),  # Use FP16 if BF16 not supported
        bf16=torch.cuda.is_bf16_supported(),      # Use BF16 if supported (RTX 30 series)
        logging_steps=10,
        optim="adamw_8bit",  # Memory-efficient optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        save_steps=500,
        save_total_limit=2,  # Only keep 2 checkpoints to save disk space
        dataloader_pin_memory=False,  # Reduce memory usage
        remove_unused_columns=False,
        push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
        report_to=None,  # Disable wandb logging to save memory
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,  # Can help with memory
        args=training_args,
    )
    
    # Monitor GPU memory before training
    if torch.cuda.is_available():
        print(f"GPU Memory before training:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.1f}GB")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training complete! Model saved to: {output_dir}")
    
    return model, trainer

def test_model(model, tokenizer, test_prompts: list = None):
    """
    Test the trained model with sample prompts
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        test_prompts: List of test prompts
    """
    if test_prompts is None:
        test_prompts = [
            "How do I handle a player who wants to attempt something not covered by the rules?",
            "What's the best way to balance combat encounters for new players?",
            "How should I interpret ambiguous spell descriptions?",
            "What do I do when players try to break the game with creative rule interpretations?"
        ]
    
    print("\n" + "="*50)
    print("Testing the trained model:")
    print("="*50)
    
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        
        messages = [
            {"role": "system", "content": "You are an expert Game Master for tabletop RPGs. Provide helpful, practical advice."},
            {"role": "user", "content": prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        
        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs, 
                streamer=text_streamer, 
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

def export_model(model, tokenizer, output_dir: str):
    """
    Export the trained model in different formats
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        output_dir: Directory containing the trained model
    """
    print(f"\nExporting model from: {output_dir}")
    
    # Export to GGUF format for efficient inference (optional)
    try:
        print("Exporting to GGUF format...")
        model.save_pretrained_gguf(
            output_dir + "_gguf", 
            tokenizer,
            quantization_method="q4_k_m"  # Good balance of quality and size
        )
        print(f"GGUF model saved to: {output_dir}_gguf")
    except Exception as e:
        print(f"GGUF export failed (this is optional): {e}")
    
    # Export to regular format for Hugging Face transformers
    try:
        print("Exporting merged model...")
        model.save_pretrained_merged(
            output_dir + "_merged",
            tokenizer,
            save_method="merged_16bit",  # or "merged_4bit" for smaller size
        )
        print(f"Merged model saved to: {output_dir}_merged")
    except Exception as e:
        print(f"Merged export failed: {e}")

def monitor_gpu_memory():
    """Monitor and display GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
        
        # Clean up if memory usage is high
        if allocated > 10:  # If using more than 10GB
            torch.cuda.empty_cache()
            gc.collect()
    else:
        print("CUDA not available")

def main():
    """Main training pipeline"""
    print("TTRPG LLM Training Script")
    print("=" * 50)
    
    # Configuration
    config = {
        "model_name": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",  # Change if needed
        "data_file": "./datasets/ttrpg_training_data_conversational.json",   # From PDF extractor
        "output_dir": "./trained_ttrpg_gm",
        "max_seq_length": 2048,
        "max_samples": None,  # Set to small number like 100 for testing
        "test_model_after_training": True
    }
    
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    try:
        # Step 1: Setup model
        print("\n1. Setting up model...")
        model, tokenizer = setup_model_for_training(
            model_name=config["model_name"],
            max_seq_length=config["max_seq_length"]
        )
        monitor_gpu_memory()
        
        # Step 2: Prepare dataset
        print("\n2. Preparing dataset...")
        dataset = prepare_dataset(
            data_file=config["data_file"],
            tokenizer=tokenizer,
            max_samples=config["max_samples"]
        )
        
        # Step 3: Train model
        print("\n3. Training model...")
        model, trainer = train_ttrpg_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_dir=config["output_dir"]
        )
        monitor_gpu_memory()
        
        # Step 4: Test model (optional)
        if config["test_model_after_training"]:
            print("\n4. Testing trained model...")
            test_model(model, tokenizer)
        
        # Step 5: Export model
        print("\n5. Exporting model...")
        export_model(model, tokenizer, config["output_dir"])
        
        print("\n" + "="*50)
        print("Training pipeline completed successfully!")
        print(f"Your TTRPG GM model is ready at: {config['output_dir']}")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Possible solutions:")
        print("1. Reduce batch_size or max_seq_length")
        print("2. Use a smaller model")
        print("3. Reduce max_samples for testing")
        print("4. Check GPU memory with nvidia-smi")
        raise

if __name__ == "__main__":
    main()
