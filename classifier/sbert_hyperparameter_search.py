import math
import logging
import os
import gc

import apex
from apex.optimizers import FusedAdam
from torch.utils.data import DataLoader, Dataset
import transformers
from sentence_transformers import (
    InputExample,
    models,
    SentenceTransformer,
    LoggingHandler,
)

# from sentence_transformers.evaluation import LabelAccuracyEvaluator

import pandas as pd
import optuna
import wandb

from constants import (
    ISOMER_DESIGN_TRAIN_DS_PATH,
    ISOMER_DESIGN_VAL_DS_PATH,
    MODEL_NAMES,
    CAT_TO_LABEL,
)
from loss import SoftmaxLoss, SelfAdjDiceLoss
from evaluator import LabelAccuracyEvaluator

os.environ["TOKENIZERS_PARALLELISM"] = "true"
apex.torch.backends.cuda.matmul.allow_tf32 = True
apex.torch.backends.cudnn.allow_tf32 = True
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


class PandasDataFrameDataset(Dataset):
    """
    PyTorch Dataset class for a Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        df = df.rename(columns={"smiles": "texts", "category": "label"})
        df["label"] = df["label"].map(CAT_TO_LABEL)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return InputExample(texts=row["texts"], label=row["label"])


def load_data(
    batch_size: int, train_file: str, val_file: str, test_file: str | None = None
):
    # Load train data
    train_df = pd.read_parquet(train_file, columns=["smiles", "category"])
    train_ds = PandasDataFrameDataset(train_df)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=16,
    )

    # Load validation data
    val_df = pd.read_parquet(val_file, columns=["smiles", "category"])
    val_ds = PandasDataFrameDataset(val_df)
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=16,
    )

    # Load test data if provided
    if test_file:
        test_df = pd.read_parquet(test_file, columns=["smiles", "category"])
        test_ds = PandasDataFrameDataset(test_df)
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            pin_memory_device="cuda",
            num_workers=16,
        )
    else:
        test_dl = None

    # Get the number of unique labels
    num_labels = train_df["category"].nunique()

    return train_dl, val_dl, test_dl, num_labels


def objective(
    trial: optuna.Trial,
) -> float:
    apex.torch.clear_autocast_cache()
    apex.torch.cuda.empty_cache()
    gc.collect()
    transformers.set_seed(42)

    # model_name = trial.suggest_categorical("model_name", MODEL_NAMES)
    model_name = (
        "/home/manny/source/chem-mrl/output/"
        # "ChemBERTa-zinc-base-v1-QED_functional_morgan_fingerprint-2d-matryoshka-embeddings"
        # "-num_epochs_2-epoch_2-best-model-1900000_steps"
        "ChemBERTa-zinc-base-v1-2d-matryoshka-embeddings"
        "-n_layers_per_step_2-TaniLoss-lr_1.1190785944700813e-05-batch_size_8"
        "-num_epochs_2-epoch_2-best-model-1900000_steps"
    )
    matryoshka_dim = 768
    # if "seyonec" not in model_name:
    #     matryoshka_dim = trial.suggest_categorical(
    #         "matryoshka_dim", [768, 512, 256, 128, 64, 32]
    #     )

    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model], truncate_dim=matryoshka_dim
    )

    # observation: longer epoch that more the search heuristic model prefers warmupcosinewithhardrestarts
    num_epochs = 3
    warmup_steps_percent = 2
    # lr_base = trial.suggest_float(
    #     "lr_base", 2.0e-06, 8e-06
    # )  # 5.6e-06 median lr_base after initial search - init best model lr_base was way lower than this
    lr_base = 3.4038386108141304e-06
    scheduler = "warmupcosinewithhardrestarts"
    # scheduler = trial.suggest_categorical(
    #     "scheduler",
    #     [
    #         "warmuplinear",
    #         "warmupcosine",
    #         # "warmupcosinewithhardrestarts",
    #     ],
    # )
    dropout_p = 0.15
    loss_func = "SoftMax"  # trial.suggest_categorical("loss_func", ["SelfAdjDice"])  # "SoftMax",
    if loss_func == "SelfAdjDice":
        dice_reduction = "mean"
        # dice_reduction = trial.suggest_categorical("dice_reduction", ["sum", "mean"])

    max_seq_length = word_embedding_model.max_seq_length
    train_batch_size = 160
    # train_batch_size = int(trial.suggest_float("train_batch_size", 296, 344, step=16))
    LR = lr_base * math.sqrt(train_batch_size)

    loss_parameter_str = (
        f"-dice_reduction_{dice_reduction}" if loss_func == "SelfAdjDice" else ""
    )
    model_save_path = (
        "output/"
        + model_name.rsplit("/", 1)[1][:20]
        + "-classification-model"
        + f"-epochs_{num_epochs}"
        + f"-mrl_dim_{matryoshka_dim}"
        + f"-lr_{lr_base}"
        + f"-batch_size_{train_batch_size}"
        + f"-warmup_steps_percent_{warmup_steps_percent}"
        + f"-scheduler_{scheduler}"
        + f"-dropout_p_{dropout_p}"
        + f"-loss_{loss_func}"
        + loss_parameter_str
    )
    print(f"\n{model_save_path}\n")

    wandb.init(
        project="chem-mrl-classification-hyperparameter-search",
        config={
            "model_name": model_name,
            "train_batch_size": train_batch_size,
            "max_seq_length": max_seq_length,
            "matryoshka_dim": matryoshka_dim,
            "num_epochs": num_epochs,
            "lr_base": lr_base,
            "warmup_steps_percent": warmup_steps_percent,
            "scheduler": scheduler,
            "loss_func": loss_func,
            "dropout_p": dropout_p,
        },
    )

    train_dataloader, val_dataloader, _, num_labels = load_data(
        batch_size=train_batch_size,
        train_file=ISOMER_DESIGN_TRAIN_DS_PATH,
        val_file=ISOMER_DESIGN_VAL_DS_PATH,
    )

    if loss_func == "SoftMax":
        train_loss = SoftmaxLoss(
            model=model,
            # method returns truncated dim if using truncated mrl model
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=num_labels,
            dropout=dropout_p,
        )
    else:
        train_loss = SelfAdjDiceLoss(
            model=model,
            # method returns truncated dim if using truncated mrl model
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=num_labels,
            reduction=dice_reduction,
            dropout=dropout_p,
        )

    val_evaluator = LabelAccuracyEvaluator(
        dataloader=val_dataloader,
        softmax_model=train_loss,
        write_csv=True,
    )

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * warmup_steps_percent)
    logging.info("Warmup-steps: {}".format(warmup_steps))

    total_number_training_points = len(train_dataloader) * train_batch_size
    weight_decay = 0.05 * math.sqrt(
        train_batch_size / (total_number_training_points * num_epochs)
    )

    def wandb_callback(score, epoch, _):
        eval_dict = {
            "accuracy": score,
            "epoch": epoch,
            "lr_base": lr_base,
            "model_name": model_name,
            "num_epochs": num_epochs,
            "matryoshka_dim": matryoshka_dim,
            "train_batch_size": train_batch_size,
            "warmup_steps_percent": warmup_steps_percent,
            "scheduler": scheduler,
            "loss_func": loss_func,
            "dropout_p": dropout_p if loss_func == "SoftMax" else None,
            "dice_reduction": dice_reduction if loss_func == "SelfAdjDice" else None,
        }
        wandb.log(eval_dict)
        apex.torch.clear_autocast_cache()
        apex.torch.cuda.empty_cache()
        gc.collect()

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        evaluation_steps=4,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        optimizer_class=FusedAdam,
        optimizer_params={
            "lr": LR,
            "weight_decay": weight_decay,
            "adam_w_mode": True,
        },
        use_amp=False,
        show_progress_bar=True,
        scheduler=scheduler,
        checkpoint_path="output",
        # checkpoint_save_steps=256,
        callback=wandb_callback,
    )

    eval_file_path = os.path.join(
        model_save_path, "eval/accuracy_evaluation_results.csv"
    )
    eval_results_df = pd.read_csv(eval_file_path)
    metric = float(eval_results_df.iloc[-1]["accuracy"])
    return metric


def generate_hyperparameters():
    study = optuna.create_study(
        study_name="chem-mrl-classification-hyperparameter-tuning",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=1,  #  512, 768, 1536
        gc_after_trial=True,
        show_progress_bar=True,
    )

    print("Best hyperparameters found:")
    print(study.best_params)
    print("Best best trials:")
    print(study.best_trials)
    print("Best trial:")
    print(study.best_trial)
    study.trials_dataframe().to_csv(
        "chem-mrl-classification-hyperparameter-tuning.csv", index=False
    )


if __name__ == "__main__":
    generate_hyperparameters()
