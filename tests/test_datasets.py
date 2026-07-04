"""Tests for dataset processor."""
import pytest
from finetuning.datasets.processor import DatasetProcessor
from finetuning.models.schemas import InstructionSample


@pytest.fixture
def processor() -> DatasetProcessor:
    return DatasetProcessor()


class TestDatasetProcessor:
    def test_format_instruction_with_input(self, processor: DatasetProcessor) -> None:
        sample = InstructionSample(instruction="Summarize", input="Long text here", output="Short summary")
        result = processor.format_instruction_sample(sample)
        assert "### Instruction:" in result
        assert "### Input:" in result
        assert "### Response:" in result

    def test_format_instruction_without_input(self, processor: DatasetProcessor) -> None:
        sample = InstructionSample(instruction="Tell a joke", input="", output="Why did the chicken...")
        result = processor.format_instruction_sample(sample)
        assert "### Instruction:" in result
        assert "### Input:" not in result

    def test_validate_instruction_dataset_valid(self, processor: DatasetProcessor) -> None:
        samples = [
            {"instruction": "Do X", "output": "Done X"},
            {"instruction": "Do Y", "output": "Done Y"},
        ]
        valid, errors = processor.validate_instruction_dataset(samples)
        assert len(valid) == 2
        assert len(errors) == 0

    def test_validate_instruction_dataset_missing_fields(self, processor: DatasetProcessor) -> None:
        samples = [
            {"instruction": "Do X"},  # Missing output
            {"output": "Done Y"},  # Missing instruction
            {"instruction": "Good", "output": "Valid"},
        ]
        valid, errors = processor.validate_instruction_dataset(samples)
        assert len(valid) == 1
        assert len(errors) == 2

    def test_validate_preference_dataset(self, processor: DatasetProcessor) -> None:
        samples = [
            {"prompt": "Q", "chosen": "Good answer", "rejected": "Bad answer"},
            {"prompt": "Q2", "chosen": "Same", "rejected": "Same"},  # Identical
        ]
        valid, errors = processor.validate_preference_dataset(samples)
        assert len(valid) == 1
        assert len(errors) == 1

    def test_split_dataset(self, processor: DatasetProcessor) -> None:
        samples = list(range(100))
        train, test = processor.split_dataset(samples, test_ratio=0.2)
        assert len(train) == 80
        assert len(test) == 20
