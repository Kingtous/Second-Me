"""Utility functions for the L2 model training and inference.

This module provides utilities for token counting, model preparation, data processing,
and other helper functions used across the L2 pipeline.
"""

from collections import defaultdict
from datetime import datetime
from enum import Enum
import json
import os
import sys

from datasets import DatasetDict, Dataset, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import tiktoken
import torch
import logging
from lpm_kernel.configs.logging import TRAIN_LOG_FILE

from lpm_kernel.L2.training_prompt import (
    CONTEXT_PROMPT,
    CONTEXT_COT_PROMPT,
    JUDGE_PROMPT,
    JUDGE_COT_PROMPT,
    MEMORY_PROMPT,
    MEMORY_COT_PROMPT,
)


# Default chat templates for different model formats
DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def count_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string using a specified encoding.

    Args:
        string: Text to tokenize.
        encoding_name: The encoding name to use. Defaults to "cl100k_base".

    Returns:
        The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_string_by_tokens(
    string: str, max_tokens: int, encoding_name: str = "cl100k_base"
) -> str:
    """Truncates a string to fit within a specified number of tokens.
    
    Args:
        string: Text to truncate.
        max_tokens: Maximum number of tokens to keep.
        encoding_name: The encoding name to use. Defaults to "cl100k_base".
        
    Returns:
        The truncated string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(string)
    if len(tokens) > max_tokens:
        # Truncate the tokens to the maximum token limit
        truncated_tokens = tokens[:max_tokens]
        # Decode the truncated tokens back to a string
        truncated_string = encoding.decode(truncated_tokens)
        return truncated_string
    return string

class ChatmlSpecialTokens(str, Enum):
    """Special tokens for ChatML format models."""
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        """Returns a list of all special tokens."""
        return [token.value for token in cls]

class ZephyrSpecialTokens(str, Enum):
    """Special tokens for Zephyr format models."""
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        """Returns a list of all special tokens."""
        return [token.value for token in cls]

def create_and_prepare_model(args, data_args, training_args):
    """Creates and prepares a model for training.
    
    Args:
        args: Model arguments containing model configuration.
        data_args: Data arguments for training.
        training_args: Training configuration arguments.
        
    Returns:
        Tuple of (model, tokenizer) ready for training.
    """
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                logging.info("=" * 80)
                logging.info(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                logging.info("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        if os.getenv("PLATFORM") != "apple":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                torch_dtype=torch.bfloat16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                trust_remote_code=True
            )

    peft_config = None
    chat_template = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
            padding_side="right",
        )
        tokenizer.chat_template = chat_template
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    return model, peft_config, tokenizer


def create_chat_data(data_args, tokenizer):
    """Creates and preprocesses chat data for training.
    
    Args:
        data_args: Arguments for dataset configuration.
        tokenizer: Tokenizer for text processing.
        
    Returns:
        Processed dataset ready for training.
    """
    def preprocess(sample, user_name='user', is_cot=False):
        """Preprocesses a chat sample.
        
        Args:
            sample: The input sample to process.
            user_name: Name of the user. Defaults to 'user'.
            is_cot: Whether to use chain-of-thought prompts. Defaults to False.
            
        Returns:
            Processed chat sample.
        """
        if sample.get('assistant') is None and sample.get('enhanced_request') is not None:
            user_message = f"{user_name}'s request is: " + sample['user_request']
            messages = [
                {"role": "system", "content": CONTEXT_COT_PROMPT.format(user_name=user_name) if is_cot else CONTEXT_PROMPT.format(user_name=user_name)},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": sample['enhanced_request'].strip('\n')},
            ]
            return [{"content": tokenizer.apply_chat_template(messages, tokenize=False)}]
        if sample.get('assistant') is None and sample.get('user_feedback') is not None:
            user_message = f"{user_name}'s request is: " + sample['user_request'] + "\n" + "Expert's response is: " + sample['expert_response']
            messages = [
                {"role": "system", "content": JUDGE_COT_PROMPT.format(user_name=user_name) if is_cot else JUDGE_PROMPT.format(user_name=user_name)},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": sample['user_feedback'].strip('\n')},
            ]
            return [{"content": tokenizer.apply_chat_template(messages, tokenize=False)}]
        
        if sample.get('assistant') is None:
            return []
        sample['assistant'] = sample['assistant'].strip('\n')
        
        messages = [
            {"role": "system", "content": MEMORY_COT_PROMPT.format(user_name=user_name) if is_cot else MEMORY_PROMPT.format(user_name=user_name)},
            {"role": "user", "content": sample['user']},
            {"role": "assistant", "content": sample['assistant']},
        ]
        if 'None' in sample['assistant']:
            return []
        return [{"content": tokenizer.apply_chat_template(messages, tokenize=False)}]
    
    dataset = load_dataset("json", data_files=data_args.dataset_name, split="train")
    res_dataset = []
    
    for case in dataset:
        res_dataset.extend(preprocess(case, data_args.user_name, data_args.is_cot))
    
    res = Dataset.from_list(res_dataset)
    print(f"**************Dataset contains {res.num_rows} elements.**************")

    return res


def formatting_prompts_func(example):
    """Format examples for training.
    
    Args:
        example: Example to format.
        
    Returns:
        Formatted text.
    """
    out_text_list = []
    for i in range(len(example["content"])):
        out_text_list.append(example["content"][i])
    return out_text_list

# Improved logging setup
def setup_logger(log_path, logger_name="download_logger"):
    """Setup a logger with file and console handlers."""
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_hf_model(model_name="Qwen2.5-0.5B-Instruct", log_file_path=None) -> str:
    """Saves a Hugging Face model locally.
    
    Args:
        model_name: Name of the model to save. Defaults to "Qwen2.5-0.5B-Instruct".
        log_file_path: Path to save download logs. If None, uses default path.
        
    Returns:
        Path to the saved model.
    """
    # If log_file_path is None or empty, use default path
    if not log_file_path:
        log_file_path = TRAIN_LOG_FILE
    
    # Setup logging
    logger = setup_logger(log_file_path)
    
    base_dir = os.path.join(os.getcwd(), "resources/L2/base_models")
    normalized_model_name = os.path.normpath(model_name)
    if ".." in normalized_model_name or normalized_model_name.startswith("/"):
        raise ValueError("Invalid model name")
    save_path = os.path.join(base_dir, normalized_model_name)
    os.makedirs(save_path, exist_ok=True)

    from huggingface_hub import list_repo_files, configure_http_backend
    import requests
    from tqdm import tqdm
    from tqdm.contrib.concurrent import thread_map
    import shutil
    
    # Set a higher timeout, but remove the unsupported pool_size parameter
    try:
        # Try using the timeout parameter to configure
        configure_http_backend(timeout=100.0)
    except TypeError:
        # If the timeout parameter is also not supported, do not use any parameters
        try:
            configure_http_backend()
        except Exception as e:
            logger.warning(f"Failed to configure HTTP backend: {e}")

    # Log download start
    logger.info(f"Starting download of model: {model_name}")
    logger.info(f"Will be saved to: {save_path}")
    
    if os.path.exists(save_path):
        logger.info(f"Model already exists at {save_path}. Skipping download.")
        return save_path
    
    hf_model_name = f"Qwen/{model_name}"
    
    try:
        # First get the list of all files in the repository
        files = list_repo_files(hf_model_name)
        logger.info(f"Found {len(files)} files to download from {hf_model_name}")
        
        # Define a function for downloading a single file and recording progress
        def download_file_with_progress(file_info):
            filename, file_path = file_info
            try:
                # Build the download URL
                url = f"https://huggingface.co/{hf_model_name}/resolve/main/{filename}"
                
                # Target file path
                local_file_path = os.path.join(save_path, filename)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Check if the file already exists
                if os.path.exists(local_file_path):
                    logger.info(f"File already exists: {filename}")
                    return filename, True
                
                # Get file size
                response = requests.head(url)
                total_size = int(response.headers.get('content-length', 0))
                
                # If the size cannot be obtained, set a default value or do not display the percentage
                if total_size == 0:
                    logger.info(f"Starting download of file: {filename} (Size unknown)")
                else:
                    logger.info(f"Starting download of file: {filename} (Size: {total_size / 1024 / 1024:.2f} MB)")
                
                # Create the file to write to
                with open(local_file_path, 'wb') as f:
                    # Create a progress bar, if the total size is unknown, set it to None
                    progress_bar = tqdm(
                        total=total_size if total_size > 0 else None,
                        unit='iB',
                        unit_scale=True,
                        desc=f"Downloading {os.path.basename(filename)}",
                        disable=False
                    )
                    
                    # Define the progress callback function
                    def progress_callback(current, total):
                        # Update the progress bar
                        progress_bar.update(current - progress_bar.n)
                        
                        # Record the log every 1MB (or a value close to 1MB)
                        if current % (1024 * 1024) < 8192:  # Record every 1MB
                            # Ensure total is greater than 0 before calculating the percentage
                            if total and total > 0:  # Use and to ensure total is not None and greater than 0
                                percent = current / total * 100
                                logger.info(f"File {filename}: Downloaded {current/1024/1024:.2f} MB / {total/1024/1024:.2f} MB ({percent:.2f}%)")
                            else:
                                # If the total size is unknown or 0, only show the downloaded size
                                logger.info(f"File {filename}: Downloaded {current/1024/1024:.2f} MB (total size unknown)")
                
                    # Use the request library to download the file and update the progress
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        downloaded = 0
                        
                        # Check if the response contains the Content-Length header information
                        actual_total = int(response.headers.get('content-length', 0))
                        if actual_total > 0 and (total_size == 0 or total_size != actual_total):
                            # If the HEAD request did not return the correct size, but the GET request did, then update the total size
                            total_size = actual_total
                            logger.info(f"Updated file size for {filename}: {total_size / 1024 / 1024:.2f} MB")
                            progress_bar.total = total_size
                            progress_bar.refresh()
                        
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:  # Filter out empty chunks that keep the connection alive
                                f.write(chunk)
                                downloaded += len(chunk)
                                progress_callback(downloaded, total_size)
                                
                        progress_bar.close()
                        logger.info(f"Completed download of file: {filename}")
                        return filename, True
                    else:
                        logger.error(f"Failed to download {filename}: HTTP status {response.status_code}")
                        return filename, False
                        
            except Exception as e:
                logger.error(f"Error downloading {filename}: {str(e)}")
                return filename, False
        
        # Create a list of file path information
        file_infos = [(filename, os.path.join(save_path, filename)) for filename in files]
        
        # Use a thread pool to download all files in parallel
        logger.info(f"Starting parallel download of {len(file_infos)} files")
        
        # Use a thread pool to download all files in parallel
        from concurrent.futures import ThreadPoolExecutor
        
        # Limit the number of concurrent requests to avoid too many requests
        max_workers = min(10, len(file_infos))
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_file = {executor.submit(download_file_with_progress, file_info): file_info[0] 
                             for file_info in file_infos}
            
            # Wait for all tasks to complete and collect results
            for future in tqdm(future_to_file, desc="Overall Progress", unit="file"):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Finished processing {filename}")
                except Exception as exc:
                    logger.error(f'{filename} generated an exception: {exc}')
                    results.append((filename, False))
        
        # Check the download results
        success_count = sum(1 for _, success in results if success)
        logger.info(f"Download completed. Successfully downloaded {success_count}/{len(files)} files.")
        
        # Record the download completion information
        try:
            import glob
            file_count = len(glob.glob(f"{save_path}/**/*", recursive=True))
            logger.info(f"Model {model_name} downloaded with {file_count} files.")
        except Exception:
            logger.info(f"Download completed for model: {model_name}.")
        
    except KeyboardInterrupt:
        logger.warning(f"Download interrupted by user for model: {model_name}")
        raise
    except Exception as e:
        # Log any errors that occur
        logger.error(f"Error downloading model: {str(e)}")
        raise
    
    return save_path


def format_timestr(utc_time_str):
    """Formats a UTC time string to a more readable format.
    
    Args:
        utc_time_str: UTC time string to format.
        
    Returns:
        Formatted time string.
    """
    # Define the original time format
    try:
        # Parse the UTC time
        utc_time = datetime.strptime(utc_time_str, "%Y-%m-%d %H:%M:%S%z")
        
        # Convert to readable format
        formatted_time = utc_time.strftime("%B %d, %Y at %I:%M %p")
        
        return formatted_time
    except ValueError:
        # Handle invalid date format
        return utc_time_str


if __name__ == "__main__":
    if len(sys.argv) > 1:
        save_hf_model(sys.argv[1])
    else:
        save_hf_model()
