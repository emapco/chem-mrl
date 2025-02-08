import logging

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader

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
        dataloader: DataLoader,
        softmax_model: torch.nn.Module,
        name: str = "",
        write_csv: bool = True,
    ):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_" + name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = ".",
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
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
        self.dataloader.collate_fn = model.smart_batching_collate
        for _, batch in enumerate(self.dataloader):
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

        return accuracy
