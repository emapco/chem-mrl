# CHEM-MRL

[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ChemMRL%20Collection-FFD21E)](https://huggingface.co/collections/Derify/chemmrl-68d5b6f6c6f34b492b119766)
![PyPI - Version](https://img.shields.io/pypi/v/chem-mrl)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/chem-mrl?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/chem-mrl)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/emapco/chem-mrl/ci.yml)
![PyPI - Status](https://img.shields.io/pypi/status/chem-mrl)

Chem-MRL is a SMILES embedding transformer model that leverages Matryoshka Representation Learning (MRL) to generate efficient, truncatable embeddings for downstream tasks such as classification, clustering, and database querying.

The model employs [SentenceTransformers' (SBERT)](https://sbert.net/) [2D Matryoshka Sentence Embeddings](https://sbert.net/examples/training/matryoshka/README.html) (`Matryoshka2dLoss`) to enable truncatable embeddings with minimal accuracy loss, improving query performance and flexibility in downstream applications.

Datasets should consists of SMILES pairs and their corresponding [Morgan fingerprint](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints) Tanimoto similarity scores.

Hyperparameter optimization indicates that a custom Tanimoto similarity loss function, [`TanimotoSentLoss`](https://github.com/emapco/chem-mrl/blob/main/chem_mrl/losses/TanimotoLoss.py), based on [CoSENTLoss](https://kexue.fm/archives/8847), outperforms CoSENTLoss, [Tanimoto similarity](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2), [AnglELoss](https://arxiv.org/pdf/2309.12871), and cosine similarity.

## Installation

### Install with pip

```bash
pip install chem-mrl
```

### Install from source code

```bash
pip install -e .
```

### Install Flash Attention (optional, improves training speed)
The default base model, `Derify/ModChemBERT-IR-BASE`, benefits from Flash Attention for faster training and inference. Install it via pip:

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

For more information and installation options, refer to the [Flash Attention repository](https://github.com/Dao-AILab/flash-attention).

## Usage

### Hydra & Training Scripts

Hydra configuration files are in `chem_mrl/conf`. The base config (`base.yaml`) defines shared arguments and includes model-specific configurations from `chem_mrl/conf/model`. Supported models: `chem_mrl`, `chem_2d_mrl`, `classifier`, and `dice_loss_classifier`.

**Training Examples:**

```bash
# Default (chem_mrl model)
python scripts/train_chem_mrl.py

# Specify model type
python scripts/train_chem_mrl.py model=chem_2d_mrl
python scripts/train_chem_mrl.py model=classifier

# Override parameters
python scripts/train_chem_mrl.py model=chem_2d_mrl training_args.num_train_epochs=5 datasets[0].train_dataset.name=/path/to/data.parquet

# Use different custom config also located in `chem_mrl/conf`
python scripts/train_chem_mrl.py --config-name=my_custom_config.yaml
```

**Configuration Options:**
- **Command line overrides:** Use `model=<type>` and parameter overrides as shown above
- **Modify base.yaml:** Edit the `- /model: chem_mrl` line in the defaults section to change the default model, or modify any other parameters directly
- **Override config file:** Use `--config-name=<config_name>` to specify a different base configuration file instead of the default `base.yaml`

### Basic Training Workflow

To train a model, initialize the configuration with dataset paths and model parameters, then pass it to `ChemMRLTrainer` for training.

```python
from sentence_transformers import SentenceTransformerTrainingArguments

from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.schemas import BaseConfig, ChemMRLConfig, DatasetConfig, SplitConfig
from chem_mrl.schemas.Enums import FieldTypeOption
from chem_mrl.trainers import ChemMRLTrainer

dataset_config = DatasetConfig(
    key="my_dataset",
    train_dataset=SplitConfig(
        name="train.parquet",
        split_key="train",
        label_cast_type=FieldTypeOption.float32,
        sample_size=1000,
    ),
    val_dataset=SplitConfig(
        name="val.parquet",
        split_key="train",  # Use "train" for local files
        label_cast_type=FieldTypeOption.float16,
        sample_size=500,
    ),
    test_dataset=SplitConfig(
        name="test.parquet",
        split_key="train",
        label_cast_type=FieldTypeOption.float16,
        sample_size=500,
    ),
    smiles_a_column_name="smiles_a",
    smiles_b_column_name="smiles_b",
    label_column_name="similarity",
)

config = BaseConfig(
    model=ChemMRLConfig(
        model_name=BASE_MODEL_NAME,  # Predefined model name - Can be any transformer model name or path that is compatible with sentence-transformers
        n_dims_per_step=3,  # Model-specific hyperparameter
        use_2d_matryoshka=True,  # Enable 2d MRL
        # Additional parameters specific to 2D MRL models
        n_layers_per_step=2,
        kl_div_weight=0.7,  # Weight for KL divergence regularization
        kl_temperature=0.5,  # Temperature parameter for KL loss
    ),
    datasets=[dataset_config],  # List of dataset configurations
    training_args=SentenceTransformerTrainingArguments("training_output"),
)

# Initialize trainer and start training
trainer = ChemMRLTrainer(config)
test_eval_metric = (
    trainer.train()
)  # Returns the test evaluation metric if a test dataset is provided.
# Otherwise returns the final validation eval metric
```

### Custom Callbacks

You can provide a list of transformers.TrainerCallback classes to execute while training.

```python
import torch
from sentence_transformers import (
    SentenceTransformerTrainingArguments,
)
from transformers import PreTrainedModel
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.schemas import BaseConfig, ChemMRLConfig, DatasetConfig, SplitConfig
from chem_mrl.schemas.Enums import FieldTypeOption
from chem_mrl.trainers import ChemMRLTrainer


# Define a callback class for logging evaluation metrics
# https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.TrainerCallback
class EvalCallback(TrainerCallback):
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel | torch.nn.Module,
        metrics: dict[str, float],
        **kwargs,
    ) -> None:
        """
        Event called after an evaluation phase.
        """
        pass


dataset_config = DatasetConfig(
    key="callback_dataset",
    train_dataset=SplitConfig(
        name="train.parquet",
        split_key="train",
        label_cast_type=FieldTypeOption.float32,
        sample_size=1000,
    ),
    val_dataset=SplitConfig(
        name="val.parquet",
        split_key="train",
        label_cast_type=FieldTypeOption.float16,
        sample_size=500,
    ),
    smiles_a_column_name="smiles_a",
    smiles_b_column_name="smiles_b",
    label_column_name="similarity",
)

config = BaseConfig(
    model=ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
    ),
    datasets=[dataset_config],
    training_args=SentenceTransformerTrainingArguments("training_output"),
)

# Train with callback
trainer = ChemMRLTrainer(config)
val_eval_metric = trainer.train(callbacks=[EvalCallback()])
```

## Classifier

This repository includes code for training a linear classifier with optional dropout regularization. The classifier categorizes substances based on SMILES and category features.

Hyperparameter tuning shows that cross-entropy loss (`softmax` option) outperforms self-adjusting dice loss in terms of accuracy, making it the preferred choice for molecular property classification.

### Usage

#### Basic Classification Training

To train a classifier, configure the model with dataset paths and column names, then initialize `ClassifierTrainer` to start training.

```python
from sentence_transformers import SentenceTransformerTrainingArguments

from chem_mrl.schemas import BaseConfig, ClassifierConfig, DatasetConfig, SplitConfig
from chem_mrl.schemas.Enums import FieldTypeOption
from chem_mrl.trainers import ClassifierTrainer

dataset_config = DatasetConfig(
    key="classification_dataset",
    train_dataset=SplitConfig(
        name="train_classification.parquet",
        split_key="train",
        label_cast_type=FieldTypeOption.float32,
        sample_size=1000,
    ),
    val_dataset=SplitConfig(
        name="val_classification.parquet",
        split_key="train",
        label_cast_type=FieldTypeOption.float16,
        sample_size=500,
    ),
    smiles_a_column_name="smiles",
    smiles_b_column_name=None,  # Not needed for classification
    label_column_name="label",
)

# Define classification training configuration
config = BaseConfig(
    model=ClassifierConfig(
        model_name="path/to/trained_mrl_model",  # Pretrained MRL model path
    ),
    datasets=[dataset_config],
    training_args=SentenceTransformerTrainingArguments("training_output"),
)

# Initialize and train the classifier
trainer = ClassifierTrainer(config)
trainer.train()
```

#### Training with Dice Loss

For imbalanced classification tasks, **Dice Loss** can improve performance by focusing on hard-to-classify samples. Below is a configuration using `DiceLossClassifierConfig`, which introduces additional hyperparameters.

```python
from sentence_transformers import SentenceTransformerTrainingArguments

from chem_mrl.schemas import BaseConfig, ClassifierConfig, DatasetConfig, SplitConfig
from chem_mrl.schemas.Enums import ClassifierLossFctOption, DiceReductionOption, FieldTypeOption
from chem_mrl.trainers import ClassifierTrainer

dataset_config = DatasetConfig(
    key="dice_loss_dataset",
    train_dataset=SplitConfig(
        name="train_classification.parquet",
        split_key="train",
        label_cast_type=FieldTypeOption.float32,
        sample_size=1000,
    ),
    val_dataset=SplitConfig(
        name="val_classification.parquet",
        split_key="train",
        label_cast_type=FieldTypeOption.float16,
        sample_size=500,
    ),
    smiles_a_column_name="smiles",
    smiles_b_column_name=None,  # Not needed for classification
    label_column_name="label",
)

# Define classification training configuration with Dice Loss
config = BaseConfig(
    model=ClassifierConfig(
        model_name="path/to/trained_mrl_model",
        loss_func=ClassifierLossFctOption.selfadjdice,
        dice_reduction=DiceReductionOption.sum,  # Reduction method for Dice Loss (e.g., 'mean' or 'sum')
        dice_gamma=1.0,  # Smoothing factor hyperparameter
    ),
    datasets=[dataset_config],
    training_args=SentenceTransformerTrainingArguments("training_output"),
)

# Initialize and train the classifier with Dice Loss
trainer = ClassifierTrainer(config)
trainer.train()
```

## References:

- Chithrananda, Seyone, et al. "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." _arXiv [Cs.LG]_, 2020. [Link](http://arxiv.org/abs/2010.09885).
- Ahmad, Walid, et al. "ChemBERTa-2: Towards Chemical Foundation Models." _arXiv [Cs.LG]_, 2022. [Link](http://arxiv.org/abs/2209.01712).
- Kusupati, Aditya, et al. "Matryoshka Representation Learning." _arXiv [Cs.LG]_, 2022. [Link](https://arxiv.org/abs/2205.13147).
- Li, Xianming, et al. "2D Matryoshka Sentence Embeddings." _arXiv [Cs.CL]_, 2024. [Link](http://arxiv.org/abs/2402.14776).
- Bajusz, Dávid, et al. "Why is the Tanimoto Index an Appropriate Choice for Fingerprint-Based Similarity Calculations?" _J Cheminform_, 7, 20 (2015). [Link](https://doi.org/10.1186/s13321-015-0069-3).
- Li, Xiaoya, et al. "Dice Loss for Data-imbalanced NLP Tasks." _arXiv [Cs.CL]_, 2020. [Link](https://arxiv.org/abs/1911.02855)
- Reimers, Nils, and Gurevych, Iryna. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing_, 2019. [Link](https://arxiv.org/abs/1908.10084).

## Citation
If you use this code or model in your research, please cite:

```bibtex
@software{cortes-2025-chem-mrl,
    author    = {Emmanuel Cortes},
    title     = {CHEM-MRL: SMILES-based Matryoshka Representation Learning Embedding Transformer},
    year      = {2025},
    publisher = {GitHub},
    howpublished = {GitHub repository},
    url       = {https://github.com/emapco/chem-mrl},
}
```
