"""
Dataset processor for preparing fine-tuning data.

Handles format conversion, validation, splitting, and template
application for instruction-tuning and preference datasets.
"""

from finetuning.models.schemas import DatasetConfig, InstructionSample, PreferenceSample
from finetuning.utils.logger import get_logger

logger = get_logger(__name__)

# Chat template for instruction format
INSTRUCTION_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# Template without input field
INSTRUCTION_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""


class DatasetProcessor:
    """
    Processes and formats datasets for fine-tuning.

    Supports instruction-tuning format, chat format, and preference
    pair format for DPO training.
    """

    def format_instruction_sample(self, sample: InstructionSample) -> str:
        """
        Format a single instruction sample into training text.

        Args:
            sample: InstructionSample with instruction, input, output.

        Returns:
            Formatted training text string.
        """
        if sample.input:
            return INSTRUCTION_TEMPLATE.format(
                instruction=sample.instruction,
                input=sample.input,
                output=sample.output,
            )
        return INSTRUCTION_TEMPLATE_NO_INPUT.format(
            instruction=sample.instruction,
            output=sample.output,
        )

    def format_chat_sample(self, messages: list[dict[str, str]]) -> str:
        """
        Format a chat conversation into training text.

        Args:
            messages: List of {role, content} message dicts.

        Returns:
            Formatted chat string.
        """
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted_parts.append(f"### System:\n{content}")
            elif role == "user":
                formatted_parts.append(f"### User:\n{content}")
            elif role == "assistant":
                formatted_parts.append(f"### Assistant:\n{content}")
        return "\n\n".join(formatted_parts)

    def validate_instruction_dataset(self, samples: list[dict]) -> tuple[list[dict], list[str]]:
        """
        Validate instruction dataset samples.

        Args:
            samples: List of raw sample dictionaries.

        Returns:
            Tuple of (valid_samples, error_messages).
        """
        valid = []
        errors = []

        for i, sample in enumerate(samples):
            if "instruction" not in sample or not sample["instruction"].strip():
                errors.append(f"Sample {i}: missing 'instruction' field")
                continue
            if "output" not in sample or not sample["output"].strip():
                errors.append(f"Sample {i}: missing 'output' field")
                continue
            valid.append(sample)

        logger.info("dataset_validated", total=len(samples), valid=len(valid), errors=len(errors))
        return valid, errors

    def validate_preference_dataset(self, samples: list[dict]) -> tuple[list[dict], list[str]]:
        """
        Validate preference dataset for DPO training.

        Args:
            samples: List of raw preference pair dictionaries.

        Returns:
            Tuple of (valid_samples, error_messages).
        """
        valid = []
        errors = []

        for i, sample in enumerate(samples):
            if "prompt" not in sample or not sample["prompt"].strip():
                errors.append(f"Sample {i}: missing 'prompt'")
                continue
            if "chosen" not in sample or not sample["chosen"].strip():
                errors.append(f"Sample {i}: missing 'chosen'")
                continue
            if "rejected" not in sample or not sample["rejected"].strip():
                errors.append(f"Sample {i}: missing 'rejected'")
                continue
            if sample["chosen"] == sample["rejected"]:
                errors.append(f"Sample {i}: 'chosen' and 'rejected' are identical")
                continue
            valid.append(sample)

        logger.info("preference_dataset_validated", total=len(samples), valid=len(valid))
        return valid, errors

    def split_dataset(self, samples: list[dict], test_ratio: float = 0.1) -> tuple[list, list]:
        """
        Split dataset into train and test sets.

        Args:
            samples: Full dataset.
            test_ratio: Fraction for test set.

        Returns:
            Tuple of (train_samples, test_samples).
        """
        split_index = int(len(samples) * (1 - test_ratio))
        train = samples[:split_index]
        test = samples[split_index:]
        logger.info("dataset_split", train_size=len(train), test_size=len(test))
        return train, test
