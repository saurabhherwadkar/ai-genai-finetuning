"""
QLoRA trainer for memory-efficient fine-tuning.

Combines 4-bit quantization (NF4) with LoRA to enable fine-tuning
of large models on consumer hardware with minimal quality loss.
"""

import uuid
from datetime import datetime, timezone

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from finetuning.config.settings import get_settings
from finetuning.models.schemas import DatasetConfig, TrainingJob, TrainingMethod, TrainingStatus
from finetuning.utils.logger import get_logger

logger = get_logger(__name__)


class QLoRATrainer:
    """
    QLoRA fine-tuning trainer.

    Loads the base model in 4-bit precision using BitsAndBytes,
    then applies LoRA adapters for parameter-efficient training.
    Enables fine-tuning 70B+ models on a single GPU.
    """

    def __init__(self) -> None:
        """Initialize QLoRA trainer."""
        self._settings = get_settings()

    def create_job(self, dataset_config: DatasetConfig) -> TrainingJob:
        """Create a new QLoRA training job."""
        job = TrainingJob(
            job_id=f"qlora-{uuid.uuid4().hex[:8]}",
            method=TrainingMethod.QLORA,
            base_model=self._settings.training.base_model,
            dataset=dataset_config,
            status=TrainingStatus.PENDING,
            output_dir=f"{self._settings.training.output_dir}/qlora-{uuid.uuid4().hex[:6]}",
        )
        logger.info("qlora_job_created", job_id=job.job_id)
        return job

    def get_quantization_config(self) -> BitsAndBytesConfig:
        """Build 4-bit quantization configuration."""
        # Map string dtype to torch dtype
        compute_dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        compute_dtype = compute_dtype_map.get(
            self._settings.qlora.bnb_4bit_compute_dtype, torch.bfloat16
        )

        return BitsAndBytesConfig(
            load_in_4bit=self._settings.qlora.load_in_4bit,
            bnb_4bit_quant_type=self._settings.qlora.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self._settings.qlora.bnb_4bit_use_double_quant,
        )

    def train(self, job: TrainingJob, dataset) -> TrainingJob:
        """
        Execute QLoRA fine-tuning with 4-bit quantized base model.

        Args:
            job: Training job configuration.
            dataset: HuggingFace dataset.

        Returns:
            Updated TrainingJob with results.
        """
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        logger.info("qlora_training_started", job_id=job.job_id)

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                job.base_model, token=self._settings.huggingface_token
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with 4-bit quantization
            bnb_config = self.get_quantization_config()
            model = AutoModelForCausalLM.from_pretrained(
                job.base_model,
                quantization_config=bnb_config,
                token=self._settings.huggingface_token,
                device_map="auto",
            )

            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)

            # Apply LoRA adapter
            lora_config = LoraConfig(
                r=self._settings.lora.rank,
                lora_alpha=self._settings.lora.alpha,
                lora_dropout=self._settings.lora.dropout,
                target_modules=self._settings.lora.target_modules,
                bias=self._settings.lora.bias,
                task_type=self._settings.lora.task_type,
            )
            model = get_peft_model(model, lora_config)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=job.output_dir,
                num_train_epochs=self._settings.training.num_epochs,
                per_device_train_batch_size=self._settings.training.batch_size,
                gradient_accumulation_steps=self._settings.training.gradient_accumulation_steps,
                learning_rate=self._settings.training.learning_rate,
                fp16=True,
                logging_steps=self._settings.training.logging_steps,
                save_steps=self._settings.training.save_steps,
                save_total_limit=2,
                report_to="none",
            )

            # Train
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                args=training_args,
                tokenizer=tokenizer,
                max_seq_length=self._settings.training.max_seq_length,
            )
            train_result = trainer.train()

            # Save adapter
            model.save_pretrained(job.output_dir)
            tokenizer.save_pretrained(job.output_dir)

            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.metrics = {"train_loss": train_result.training_loss}
            logger.info("qlora_training_completed", job_id=job.job_id)

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)
            logger.error("qlora_training_failed", error=str(e))

        return job
