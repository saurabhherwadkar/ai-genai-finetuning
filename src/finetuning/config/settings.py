"""Settings for the fine-tuning framework."""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class TrainingSettings(BaseSettings):
    """Training hyperparameters."""
    base_model: str = Field(default="meta-llama/Llama-3.1-8B-Instruct")
    output_dir: str = Field(default="outputs")
    num_epochs: int = Field(default=3)
    batch_size: int = Field(default=4)
    learning_rate: float = Field(default=2e-4)
    max_seq_length: int = Field(default=2048)
    gradient_accumulation_steps: int = Field(default=4)
    warmup_ratio: float = Field(default=0.03)
    weight_decay: float = Field(default=0.001)
    fp16: bool = Field(default=True)
    logging_steps: int = Field(default=10)
    save_steps: int = Field(default=100)


class LoRASettings(BaseSettings):
    """LoRA adapter configuration."""
    rank: int = Field(default=16)
    alpha: int = Field(default=32)
    dropout: float = Field(default=0.05)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bias: str = Field(default="none")
    task_type: str = Field(default="CAUSAL_LM")


class QLoRASettings(BaseSettings):
    """QLoRA quantization settings."""
    load_in_4bit: bool = Field(default=True)
    bnb_4bit_quant_type: str = Field(default="nf4")
    bnb_4bit_compute_dtype: str = Field(default="bfloat16")
    bnb_4bit_use_double_quant: bool = Field(default=True)


class DPOSettings(BaseSettings):
    """DPO training configuration."""
    beta: float = Field(default=0.1)
    learning_rate: float = Field(default=5e-5)
    max_prompt_length: int = Field(default=512)
    max_length: int = Field(default=1024)


class EvaluationSettings(BaseSettings):
    """Evaluation configuration."""
    metrics: list[str] = Field(default_factory=lambda: ["accuracy", "f1", "bleu", "rouge"])
    test_split_ratio: float = Field(default=0.1)
    num_samples: int = Field(default=100)


class APISettings(BaseSettings):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)


class LoggingSettings(BaseSettings):
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    file: str = Field(default="logs/app.log")


class Settings(BaseSettings):
    """Root settings."""
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    lora: LoRASettings = Field(default_factory=LoRASettings)
    qlora: QLoRASettings = Field(default_factory=QLoRASettings)
    dpo: DPOSettings = Field(default_factory=DPOSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    huggingface_token: str = Field(default="")

    model_config = {"env_prefix": "", "env_nested_delimiter": "__"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings."""
    env = os.getenv("APP_ENV", "development")
    config_dir = Path(__file__).parent.parent.parent.parent / "config"
    env_map = {"development": "dev", "production": "prod"}
    suffix = env_map.get(env, "")
    config_file = config_dir / f"application-{suffix}.yaml" if suffix else config_dir / "application.yaml"
    if not config_file.exists():
        config_file = config_dir / "application.yaml"
    cfg = {}
    if config_file.exists():
        with open(config_file) as f:
            cfg = yaml.safe_load(f) or {}
    return Settings(
        training=TrainingSettings(**cfg.get("training", {})) if cfg.get("training") else TrainingSettings(),
        lora=LoRASettings(**cfg.get("lora", {})) if cfg.get("lora") else LoRASettings(),
        qlora=QLoRASettings(**cfg.get("qlora", {})) if cfg.get("qlora") else QLoRASettings(),
        dpo=DPOSettings(**cfg.get("dpo", {})) if cfg.get("dpo") else DPOSettings(),
        evaluation=EvaluationSettings(**cfg.get("evaluation", {})) if cfg.get("evaluation") else EvaluationSettings(),
        api=APISettings(**cfg.get("api", {})) if cfg.get("api") else APISettings(),
        logging=LoggingSettings(**cfg.get("logging", {})) if cfg.get("logging") else LoggingSettings(),
    )
