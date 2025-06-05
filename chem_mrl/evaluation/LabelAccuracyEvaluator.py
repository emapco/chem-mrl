import logging

import torch
from datasets import Dataset
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.util import batch_to_device

from .utils import _write_results_to_csv

logger = logging.getLogger(__name__)


class LabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(
        self,
        dataset: Dataset,
        softmax_model: torch.nn.Module,
        name: str = "",
        write_csv: bool = True,
        batch_size: int = 32,
        smiles_column_name: str = "smiles",
        label_column_name: str = "label",
    ):
        """
        Constructs an evaluator for the given dataset

        Args:
            dataset (Dataset): the data for the evaluation
            softmax_model (torch.nn.Module): the softmax model for classification
            name (str): name for the evaluator
            write_csv (bool): whether to write results to CSV
            batch_size (int): batch size for evaluation
        """
        super().__init__()
        self.dataset = dataset
        self.name = name
        self.softmax_model = softmax_model
        self.batch_size = batch_size
        self.smiles_column_name = smiles_column_name
        self.label_column_name = label_column_name

        if name:
            name = "_" + name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]
        self.primary_metric = "accuracy"

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = ".",
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        model.eval()
        total = 0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Evaluation on the " + self.name + " dataset" + out_txt)

        # Process dataset in batches directly
        for i in range(0, len(self.dataset), self.batch_size):
            batch_data = self.dataset[i : i + self.batch_size]
            examples = [
                InputExample(texts=text, label=label)
                for text, label in zip(
                    batch_data[self.smiles_column_name],
                    batch_data[self.label_column_name],
                    strict=False,
                )
            ]
            batch = model.smart_batching_collate(examples)

            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)

            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()

        accuracy = correct / total

        logger.info(f"Accuracy: {accuracy:.5f} ({correct}/{total})\n")

        _write_results_to_csv(
            self.write_csv,
            self.csv_file,
            self.csv_headers,
            output_path,
            results=[epoch, steps, accuracy],
        )

        self.primary_metric = "accuracy"
        metrics = {"accuracy": accuracy}
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
