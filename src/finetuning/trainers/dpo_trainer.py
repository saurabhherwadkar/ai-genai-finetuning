"""
DPO (Direct Preference Optimization) trainer.

Aligns model outputs with human preferences without requiring a separate
reward model, using pairs of preferred/rejected responses.
"""

import uuid
from datetime import datetime, timezone

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer as TRLDPOTrainer

from finetuning.config.settings import get_settings
from finetuning.models.schemas import DatasetConfig, TrainingJob, TrainingMethod, TrainingStatus
from finetuning.utils.logger import get_logger

logger = get_logger(__name__)


class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    Trains the model to prefer chosen responses over rejected ones
    using a contrastive loss, without needing a reward model.
    """

    def __init__(self) -> None:
        """Initialize DPO trainer."""
        self._settings = get_settings()

    def create_job(self, dataset_config: DatasetConfig) -> TrainingJob:
        """Create a new DPO training job."""
        job = TrainingJob(
            job_id=f"dpo-{uuid.uuid4().hex[:8]}",
            method=TrainingMethod.DPO,
            base_model=self._settings.training.base_model,
            dataset=dataset_config,
            status=TrainingStatus.PENDING,
            output_dir=f"{self._settings.training.output_dir}/dpo-{uuid.uuid4().hex[:6]}",
        )
        logger.info("dpo_job_created", job_id=job.job_id)
        return job

    def train(self, job: TrainingJob, dataset) -> TrainingJob:
        """
        Execute DPO training with preference pairs.

        Dataset must have columns: 'prompt', 'chosen', 'rejected'.

        Args:
            job: Training job configuration.
            dataset: HuggingFace dataset with preference pairs.

        Returns:
            Updated TrainingJob.
        """
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        logger.info("dpo_training_started", job_id=job.job_id)

        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                job.base_model, token=self._settings.huggingface_token
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                job.base_model,
                token=self._settings.huggingface_token,
                torch_dtype="auto",
                device_map="auto",
            )

            # LoRA config for DPO (parameter-efficient)
            peft_config = LoraConfig(
                r=self._settings.lora.rank,
                lora_alpha=self._settings.lora.alpha,
                lora_dropout=self._settings.lora.dropout,
                target_modules=self._settings.lora.target_modules,
                bias=self._settings.lora.bias,
                task_type=self._settings.lora.task_type,
            )

            # DPO training config
            dpo_config = DPOConfig(
                output_dir=job.output_dir,
                num_train_epochs=self._settings.training.num_epochs,
                per_device_train_batch_size=self._settings.training.batch_size,
                learning_rate=self._settings.dpo.learning_rate,
                beta=self._settings.dpo.beta,
                max_prompt_length=self._settings.dpo.max_prompt_length,
                max_length=self._settings.dpo.max_length,
                logging_steps=self._settings.training.logging_steps,
                save_steps=self._settings.training.save_steps,
                report_to="none",
            )

            # Initialize DPO trainer
            trainer = TRLDPOTrainer(
                model=model,
                args=dpo_config,
                train_dataset=dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
            )

            # Train
            train_result = trainer.train()

            # Save
            trainer.save_model(job.output_dir)
            tokenizer.save_pretrained(job.output_dir)

            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.metrics = {"train_loss": train_result.training_loss}
            logger.info("dpo_training_completed", job_id=job.job_id)

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)
            logger.error("dpo_training_failed", error=str(e))

        return job
