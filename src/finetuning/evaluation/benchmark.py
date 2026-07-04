"""
Model benchmarking for comparing fine-tuned vs base vs prompted models.

Evaluates models on accuracy, F1, BLEU, and ROUGE metrics
to determine if fine-tuning improved over the baseline.
"""

import time

from finetuning.models.schemas import EvalResult
from finetuning.utils.logger import get_logger

logger = get_logger(__name__)


class ModelBenchmark:
    """
    Benchmarks and compares model performance.

    Runs evaluation samples through multiple models and reports
    comparative metrics to determine fine-tuning effectiveness.
    """

    def __init__(self) -> None:
        """Initialize the benchmark runner."""
        self._results: list[EvalResult] = []

    def evaluate_accuracy(self, predictions: list[str], references: list[str]) -> float:
        """
        Calculate exact-match accuracy.

        Args:
            predictions: Model predictions.
            references: Ground truth answers.

        Returns:
            Accuracy score 0.0 to 1.0.
        """
        if not predictions:
            return 0.0
        correct = sum(
            1 for pred, ref in zip(predictions, references)
            if pred.strip().lower() == ref.strip().lower()
        )
        return correct / len(predictions)

    def evaluate_f1(self, predictions: list[str], references: list[str]) -> float:
        """
        Calculate token-level F1 score averaged across samples.

        Args:
            predictions: Model predictions.
            references: Ground truth answers.

        Returns:
            Average F1 score.
        """
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())

            if not pred_tokens and not ref_tokens:
                f1_scores.append(1.0)
                continue
            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue

            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)

            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))

        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    def evaluate_bleu(self, predictions: list[str], references: list[str]) -> float:
        """
        Calculate simplified BLEU score (unigram precision).

        Args:
            predictions: Model predictions.
            references: Ground truth.

        Returns:
            Average BLEU-1 score.
        """
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = set(ref.lower().split())

            if not pred_tokens:
                scores.append(0.0)
                continue

            matches = sum(1 for t in pred_tokens if t in ref_tokens)
            scores.append(matches / len(pred_tokens))

        return sum(scores) / len(scores) if scores else 0.0

    def run_benchmark(
        self, model_name: str, predictions: list[str], references: list[str]
    ) -> EvalResult:
        """
        Run full benchmark suite for a model.

        Args:
            model_name: Name of the model being evaluated.
            predictions: Model outputs.
            references: Ground truth.

        Returns:
            EvalResult with all metric scores.
        """
        start_time = time.time()

        metrics = {
            "accuracy": self.evaluate_accuracy(predictions, references),
            "f1": self.evaluate_f1(predictions, references),
            "bleu": self.evaluate_bleu(predictions, references),
        }

        eval_time = (time.time() - start_time) * 1000

        result = EvalResult(
            model_name=model_name,
            metrics=metrics,
            num_samples=len(predictions),
            evaluation_time_ms=round(eval_time, 2),
        )

        self._results.append(result)
        logger.info("benchmark_complete", model=model_name, metrics=metrics)
        return result

    def compare_models(self) -> dict:
        """
        Compare all benchmarked models side by side.

        Returns:
            Dictionary with comparison results and winner per metric.
        """
        if not self._results:
            return {"error": "No benchmark results available."}

        comparison = {"models": {}, "winners": {}}

        for result in self._results:
            comparison["models"][result.model_name] = result.metrics

        # Determine winner per metric
        all_metrics = set()
        for r in self._results:
            all_metrics.update(r.metrics.keys())

        for metric in all_metrics:
            best_model = max(self._results, key=lambda r: r.metrics.get(metric, 0))
            comparison["winners"][metric] = best_model.model_name

        return comparison

    def get_results(self) -> list[EvalResult]:
        """Return all benchmark results."""
        return self._results
