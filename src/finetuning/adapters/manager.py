"""
Adapter manager for saving, loading, merging, and publishing LoRA adapters.
"""

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetuning.config.settings import get_settings
from finetuning.utils.logger import get_logger

logger = get_logger(__name__)


class AdapterManager:
    """
    Manages LoRA adapters: load, merge into base model, and save.
    """

    def __init__(self) -> None:
        """Initialize adapter manager."""
        self._settings = get_settings()

    def list_adapters(self, output_dir: str | None = None) -> list[dict]:
        """
        List available adapters in the output directory.

        Args:
            output_dir: Directory to scan. Defaults to config output_dir.

        Returns:
            List of adapter info dictionaries.
        """
        base_dir = Path(output_dir or self._settings.training.output_dir)
        adapters = []

        if not base_dir.exists():
            return adapters

        for path in base_dir.iterdir():
            if path.is_dir() and (path / "adapter_config.json").exists():
                adapters.append({
                    "name": path.name,
                    "path": str(path),
                    "has_config": True,
                })

        logger.info("adapters_listed", count=len(adapters))
        return adapters

    def merge_adapter(self, base_model: str, adapter_path: str, output_path: str) -> str:
        """
        Merge a LoRA adapter into the base model weights.

        Produces a standalone model with adapter weights baked in,
        suitable for deployment without PEFT dependency.

        Args:
            base_model: Base model identifier.
            adapter_path: Path to the saved adapter.
            output_path: Path to save the merged model.

        Returns:
            Path to the saved merged model.
        """
        logger.info("merging_adapter", base_model=base_model, adapter=adapter_path)

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=self._settings.huggingface_token,
            torch_dtype="auto",
            device_map="auto",
        )

        # Load adapter on top
        model = PeftModel.from_pretrained(model, adapter_path)

        # Merge adapter weights into base
        model = model.merge_and_unload()

        # Save merged model
        model.save_pretrained(output_path)

        # Save tokenizer alongside
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=self._settings.huggingface_token)
        tokenizer.save_pretrained(output_path)

        logger.info("adapter_merged", output=output_path)
        return output_path

    def get_adapter_info(self, adapter_path: str) -> dict:
        """
        Get configuration info for a saved adapter.

        Args:
            adapter_path: Path to the adapter directory.

        Returns:
            Dictionary with adapter configuration details.
        """
        import json

        config_path = Path(adapter_path) / "adapter_config.json"
        if not config_path.exists():
            return {"error": "No adapter_config.json found"}

        with open(config_path) as f:
            config = json.load(f)

        return {
            "path": adapter_path,
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "target_modules": config.get("target_modules"),
            "task_type": config.get("task_type"),
            "base_model": config.get("base_model_name_or_path"),
        }
