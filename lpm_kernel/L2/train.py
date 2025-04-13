from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from tqdm import tqdm
import functools
# Standard library imports
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import torch.amp
# Third-party imports
import datasets
import psutil
import torch.multiprocessing as mp
import transformers
from peft import LoraConfig
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments, set_seed
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lpm_kernel.L2.utils import (
    create_and_prepare_model,
    formatting_prompts_func,
    create_chat_data,
)
from lpm_kernel.configs.logging import LOGGING_CONFIG
import logging.config
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


# Configure how tqdm displays in logs
class LogTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 1.0)
        kwargs.setdefault("ascii", True)
        super().__init__(*args, **kwargs)

# Replace the default tqdm
sys.modules["tqdm"].tqdm = LogTqdm


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="float32",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, appends `eos_token_id` at the end of each sample being packed."
        },
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tokenizers adds special tokens to each sample being packed."
        },
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    is_sequential: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the dataset is sequential."},
    )
    is_cot: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the dataset is COT dataset."},
    )
    user_name: Optional[str] = field(
        default="User",
        metadata={"help": "The name of the user."},
    )


def main(model_args, data_args, training_args):
    logger.info(f"Python version--------------------: {sys.version}")

    # Configure logging
    logging.config.dictConfig(LOGGING_CONFIG)

    logger.info("Begin training...")

    # Ensure logs are flushed immediately
    for handler in logging.getLogger().handlers:
        handler.flush()

    logger.info("start 1")
    set_seed(training_args.seed)
    logger.info("start 2")
    # model
    model, peft_config, tokenizer = create_and_prepare_model(
        model_args, data_args, training_args
    )
    logger.info("start 3")
    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = (
        training_args.gradient_checkpointing and not model_args.use_unsloth
    )
    logger.info("start 4")
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }

    # Configure system resources for optimal performance
    def configure_system_resources(num_cores=None):
        """
        Configure system resources to optimize training performance
        
        Args:
            num_cores: Number of CPU cores to use, if None, automatically detect
        """
        # Automatically detect available cores, if not specified
        if num_cores is None:
            num_cores = min(os.cpu_count(), 6)  # Limit to 6 cores, match Docker configuration
        
        logger.info(f"Configuring system to use {num_cores} CPU cores")
        
        # Set environment variables
        os.environ["OMP_NUM_THREADS"] = str(num_cores)
        os.environ["MKL_NUM_THREADS"] = str(num_cores)
        os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)
        
        # Set PyTorch thread count
        torch.set_num_threads(num_cores)
        
        # If supported, set PyTorch multi-thread optimization
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(num_cores)
        
        # Enable memory-optimized garbage collection
        # import gc
        # gc.enable()
        
        # # Monitor memory usage and clean up periodically
        # def schedule_gc():
        #     gc.collect()
        #     torch.cuda.empty_cache() if torch.cuda.is_available() else None
        #     return schedule_gc
        
        # If CUDA is available, set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            logger.info(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
            # Display CUDA memory information
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Call function to configure system resources
    configure_system_resources()

    # datasets
    train_dataset = create_chat_data(
        data_args,
        tokenizer,
    )

    response_template = "\n<|im_start|>assistant\n"

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    training_args.dataset_kwargs = {
        "append_concat_token": data_args.append_concat_token,
        "add_special_tokens": data_args.add_special_tokens,
    }

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    logger.info("start 6")
    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        logger.info("start 6.1")
        checkpoint = training_args.resume_from_checkpoint
    logger.info("start 6.2")

    class DebugCallback(transformers.TrainerCallback):
        """
        Debug callback to monitor training process
        """

        def __init__(self):
            self.step_times = {}
            self.current_step_start = None

        def on_train_begin(self, args, state, control, **kwargs):
            logger.info("=== Training Begin ===")
            logger.info("Checking initial conditions:")
            trainer = kwargs.get("trainer")
            if trainer:
                # Check model status
                logger.info(f"Model device: {trainer.model.device}")
                logger.info(f"Model dtype: {next(trainer.model.parameters()).dtype}")

                # Check data loader
                if hasattr(trainer, "train_dataset"):
                    logger.info(f"Training dataset size: {len(trainer.train_dataset)}")

                # Check optimizer
                if hasattr(trainer, "optimizer"):
                    logger.info("Optimizer configuration:")
                    for i, group in enumerate(trainer.optimizer.param_groups):
                        logger.info(
                            f"Group {i}: lr={group['lr']}, weight_decay={group['weight_decay']}"
                        )

        def on_step_begin(self, args, state, control, **kwargs):
            self.current_step_start = time.time()
            logger.info(f"\n=== Starting Step {state.global_step + 1} ===")

            # Check system status every 10 steps
            if state.global_step % 10 == 0:
                process = psutil.Process()
                with process.oneshot():
                    logger.info(f"CPU Usage: {process.cpu_percent()}%")
                    logger.info(
                        f"Memory Usage: {process.memory_info().rss / 1024**2:.2f}MB"
                    )
                    logger.info(f"Thread Count: {process.num_threads()}")

        def on_step_end(self, args, state, control, **kwargs):
            if self.current_step_start:
                step_time = time.time() - self.current_step_start
                self.step_times[state.global_step] = step_time
                avg_time = sum(self.step_times.values()) / len(self.step_times)
                logger.info(
                    f"Step {state.global_step + 1} completed in {step_time:.2f}s (avg: {avg_time:.2f}s)"
                )

                # Check if step time is much longer than average
                if step_time > avg_time * 2 and len(self.step_times) > 1:
                    logger.warning(
                        f"Step {state.global_step + 1} took {step_time:.2f}s, which is much longer than average!"
                    )

                trainer = kwargs.get("trainer")
                if trainer and hasattr(trainer, "optimizer"):
                    # Check gradient status
                    grad_norms = []
                    for name, param in trainer.model.named_parameters():
                        if param.grad is not None:
                            grad_norms.append(param.grad.norm().item())

                    if grad_norms:
                        avg_grad_norm = sum(grad_norms) / len(grad_norms)
                        logger.info(f"Average gradient norm: {avg_grad_norm:.5f}")
                    else:
                        logger.warning("No gradients found in this step!")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                logger.info(f"=== Logs for Step {state.global_step} ===")
                for key, value in logs.items():
                    logger.info(f"{key}: {value}")

        def on_train_end(self, args, state, control, **kwargs):
            logger.info("=== Training Ended ===")
            logger.info(f"Total steps completed: {state.global_step}")
            if self.step_times:
                avg_time = sum(self.step_times.values()) / len(self.step_times)
                logger.info(f"Average step time: {avg_time:.2f}s")

    trainer.add_callback(DebugCallback())

    # Add more detailed logs
    logger.info("Starting training preparation...")
    try:
        logger.info("Initializing training process...")
        # Check model loading and structure
        logger.info("Analyzing model structure...")
        model = trainer.model

        def print_model_structure(model, prefix=""):
            logger.info(f"{prefix}Model class: {model.__class__.__name__}")
            for name, child in model.named_children():
                logger.info(f"{prefix}Child: {name} ({child.__class__.__name__})")
                if len(list(child.named_children())) > 0:
                    print_model_structure(child, prefix + "  ")

        # print_model_structure(model)

        # Check model size
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Check optimizer settings
        logger.info("Checking optimizer settings...")

        # Check data loader
        train_dataloader = trainer.get_train_dataloader()
        logger.info(f"Train dataloader created with {len(train_dataloader)} batches")

        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"Memory usage details:")
        logger.info(f"RSS (Resident Set Size): {memory_info.rss / 1024**2:.2f}MB")
        logger.info(f"VMS (Virtual Memory Size): {memory_info.vms / 1024**2:.2f}MB")

        # Start training
        logger.info("Starting actual training process...")
        trainer.train(resume_from_checkpoint=checkpoint)
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    logger.info("start 7")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    logger.info("Training completed successfully")


# Create a patch to handle autocast compatibility
def get_autocast():
    if hasattr(torch.cpu, "amp") and hasattr(torch.cpu.amp, "autocast"):
        # Old version
        return torch.cpu.amp.autocast
    else:
        # New version
        return lambda **kwargs: torch.amp.autocast("cpu", **kwargs)


# Replace the original torch.cpu.amp.autocast with our compatible function
torch.cpu.amp.autocast = get_autocast()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
