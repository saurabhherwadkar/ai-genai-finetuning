"""
LoRA (Low-Rank Adaptation) trainer for parameter-efficient fine-tuning.

Freezes the base model and trains small rank-decomposition matrices
on target attention layers, achieving near-full-finetune quality
with a fraction of the parameters.
"""

import uuid
from datetime import datetime, timezone

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from finetuning.config.settings import get_settings
from finetuning.models.schemas import DatasetConfig, TrainingJob, TrainingMethod, TrainingStatus
from finetuning.utils.logger import get_logger

logger = get_logger(__name__)


class LoRATrainer:
    """
    LoRA fine-tuning trainer.

    Applies Low-Rank Adaptation to specified model layers,
    training only the adapter weights while keeping the base frozen.
    """

    def __init__(self) -> None:
        """Initialize LoRA trainer with settings."""
        self._settings = get_settings()

    def create_job(self, dataset_config: DatasetConfig) -> TrainingJob:
        """
        Create a new LoRA training job.

        Args:
            dataset_config: Configuration for the training dataset.

        Returns:
            TrainingJob with pending status.
        """
        job = TrainingJob(
            job_id=f"lora-{uuid.uuid4().hex[:8]}",
            method=TrainingMethod.LORA,
            base_model=self._settings.training.base_model,
            dataset=dataset_config,
            status=TrainingStatus.PENDING,
            output_dir=f"{self._settings.training.output_dir}/lora-{uuid.uuid4().hex[:6]}",
        )
        logger.info("lora_job_created", job_id=job.job_id)
        return job

    def get_lora_config(self) -> LoraConfig:
        """
        Build the LoRA configuration from settings.

        Returns:
            PEFT LoraConfig object.
        """
        return LoraConfig(
            r=self._settings.lora.rank,
            lora_alpha=self._settings.lora.alpha,
            lora_dropout=self._settings.lora.dropout,
            target_modules=self._settings.lora.target_modules,
            bias=self._settings.lora.bias,
            task_type=self._settings.lora.task_type,
        )

    def get_training_arguments(self, output_dir: str) -> TrainingArguments:
        """
        Build training arguments from settings.

        Args:
            output_dir: Directory to save checkpoints.

        Returns:
            HuggingFace TrainingArguments.
        """
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self._settings.training.num_epochs,
            per_device_train_batch_size=self._settings.training.batch_size,
            gradient_accumulation_steps=self._settings.training.gradient_accumulation_steps,
            learning_rate=self._settings.training.learning_rate,
            warmup_ratio=self._settings.training.warmup_ratio,
            weight_decay=self._settings.training.weight_decay,
            fp16=self._settings.training.fp16,
            logging_steps=self._settings.training.logging_steps,
            save_steps=self._settings.training.save_steps,
            save_total_limit=3,
            report_to="none",
        )

    def train(self, job: TrainingJob, dataset) -> TrainingJob:
        """
        Execute LoRA fine-tuning.

        Args:
            job: The training job to execute.
            dataset: HuggingFace dataset object.

        Returns:
            Updated TrainingJob with results.
        """
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        logger.info("lora_training_started", job_id=job.job_id, model=job.base_model)

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                job.base_model, token=self._settings.huggingface_token
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                job.base_model,
                token=self._settings.huggingface_token,
                torch_dtype="auto",
                device_map="auto",
            )

            # Apply LoRA adapter
            lora_config = self.get_lora_config()
            model = get_peft_model(model, lora_config)

            # Log trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                "lora_params",
                trainable=trainable_params,
                total=total_params,
                percentage=round(trainable_params / total_params * 100, 2),
            )

            # Configure trainer
            training_args = self.get_training_arguments(job.output_dir)
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                args=training_args,
                tokenizer=tokenizer,
                max_seq_length=self._settings.training.max_seq_length,
            )

            # Run training
            train_result = trainer.train()

            # Save the adapter
            model.save_pretrained(job.output_dir)
            tokenizer.save_pretrained(job.output_dir)

            # Update job with metrics
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.metrics = {
                "train_loss": train_result.training_loss,
                "trainable_params": trainable_params,
                "total_params": total_params,
            }
            logger.info("lora_training_completed", job_id=job.job_id, loss=train_result.training_loss)

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)
            logger.error("lora_training_failed", job_id=job.job_id, error=str(e))

        return job
