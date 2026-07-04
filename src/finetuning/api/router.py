"""FastAPI router for fine-tuning operations."""

from fastapi import APIRouter, HTTPException

from finetuning.adapters.manager import AdapterManager
from finetuning.datasets.processor import DatasetProcessor
from finetuning.evaluation.benchmark import ModelBenchmark
from finetuning.models.schemas import DatasetConfig, EvalResult, TrainingJob
from finetuning.trainers import DPOTrainer, LoRATrainer, QLoRATrainer
from finetuning.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/finetuning", tags=["finetuning"])

_lora = LoRATrainer()
_qlora = QLoRATrainer()
_dpo = DPOTrainer()
_datasets = DatasetProcessor()
_benchmark = ModelBenchmark()
_adapters = AdapterManager()


@router.post("/jobs/lora", response_model=TrainingJob)
async def create_lora_job(dataset: DatasetConfig) -> TrainingJob:
    """Create a LoRA fine-tuning job."""
    return _lora.create_job(dataset)


@router.post("/jobs/qlora", response_model=TrainingJob)
async def create_qlora_job(dataset: DatasetConfig) -> TrainingJob:
    """Create a QLoRA fine-tuning job."""
    return _qlora.create_job(dataset)


@router.post("/jobs/dpo", response_model=TrainingJob)
async def create_dpo_job(dataset: DatasetConfig) -> TrainingJob:
    """Create a DPO training job."""
    return _dpo.create_job(dataset)


@router.post("/evaluate", response_model=EvalResult)
async def evaluate_model(model_name: str, predictions: list[str], references: list[str]) -> EvalResult:
    """Run benchmark evaluation on model outputs."""
    return _benchmark.run_benchmark(model_name, predictions, references)


@router.get("/evaluate/compare")
async def compare_models() -> dict:
    """Compare all benchmarked models."""
    return _benchmark.compare_models()


@router.get("/adapters")
async def list_adapters() -> list[dict]:
    """List available saved adapters."""
    return _adapters.list_adapters()


@router.get("/adapters/{adapter_name}/info")
async def get_adapter_info(adapter_name: str) -> dict:
    """Get adapter configuration details."""
    from finetuning.config.settings import get_settings
    settings = get_settings()
    path = f"{settings.training.output_dir}/{adapter_name}"
    return _adapters.get_adapter_info(path)


@router.get("/health")
async def health() -> dict:
    """Health check."""
    return {"status": "healthy", "service": "finetuning"}
