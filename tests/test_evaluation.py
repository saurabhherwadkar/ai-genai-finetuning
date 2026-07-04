"""Tests for model benchmarking."""
import pytest
from finetuning.evaluation.benchmark import ModelBenchmark


@pytest.fixture
def benchmark() -> ModelBenchmark:
    return ModelBenchmark()


class TestModelBenchmark:
    def test_accuracy_perfect(self, benchmark: ModelBenchmark) -> None:
        preds = ["hello", "world"]
        refs = ["hello", "world"]
        assert benchmark.evaluate_accuracy(preds, refs) == 1.0

    def test_accuracy_partial(self, benchmark: ModelBenchmark) -> None:
        preds = ["hello", "wrong"]
        refs = ["hello", "world"]
        assert benchmark.evaluate_accuracy(preds, refs) == 0.5

    def test_f1_perfect(self, benchmark: ModelBenchmark) -> None:
        preds = ["the quick brown fox"]
        refs = ["the quick brown fox"]
        assert benchmark.evaluate_f1(preds, refs) == 1.0

    def test_f1_partial_overlap(self, benchmark: ModelBenchmark) -> None:
        preds = ["the quick red fox"]
        refs = ["the quick brown fox"]
        score = benchmark.evaluate_f1(preds, refs)
        assert 0.5 < score < 1.0

    def test_bleu_perfect(self, benchmark: ModelBenchmark) -> None:
        preds = ["hello world"]
        refs = ["hello world"]
        assert benchmark.evaluate_bleu(preds, refs) == 1.0

    def test_run_benchmark(self, benchmark: ModelBenchmark) -> None:
        preds = ["Paris", "Blue"]
        refs = ["Paris", "Red"]
        result = benchmark.run_benchmark("test-model", preds, refs)
        assert result.model_name == "test-model"
        assert "accuracy" in result.metrics
        assert "f1" in result.metrics
        assert result.num_samples == 2

    def test_compare_models(self, benchmark: ModelBenchmark) -> None:
        benchmark.run_benchmark("model-a", ["yes", "no"], ["yes", "no"])
        benchmark.run_benchmark("model-b", ["yes", "wrong"], ["yes", "no"])
        comparison = benchmark.compare_models()
        assert "winners" in comparison
        assert comparison["winners"]["accuracy"] == "model-a"
