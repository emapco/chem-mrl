# CHEM-MRL

Chem-MRL is a SMILES embedding transformer model that leverages Matryoshka Representation Learning (MRL) to generate efficient, truncatable embeddings for downstream tasks such as classification, clustering, and database querying.

The model employs [SentenceTransformers' (SBERT)](https://sbert.net/) [2D Matryoshka Sentence Embeddings](https://sbert.net/examples/training/matryoshka/README.html) (`Matryoshka2dLoss`) to enable truncatable embeddings with minimal accuracy loss, improving query performance and flexibility in downstream applications.

Datasets should consists of SMILES pairs and their corresponding [Morgan fingerprint](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints) Tanimoto similarity scores.

Hyperparameter tuning indicates that a custom Tanimoto similarity loss function, [`TanimotoSentLoss`](https://github.com/emapco/chem-mrl/blob/main/chem_mrl/losses/TanimotoLoss.py), based on [CoSENTLoss](https://kexue.fm/archives/8847), outperforms [Tanimoto similarity](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2), CoSENTLoss, [AnglELoss](https://arxiv.org/pdf/2309.12871), and cosine similarity.

## v0.6.0 Release
Chem-MRL library is now built on `sentence-transformers` [v4.0.0 API](https://github.com/UKPLab/sentence-transformers/releases/tag/v4.0.1) and uses [datasets library](https://huggingface.co/docs/datasets/en/index) for loading data from local files or [Hugging Face Hub](https://huggingface.co). `sentence-transformers` library supports checkpoint resuming and extends [`transformers.Trainer`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) and [`transformers.TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) for enhanced training capabilities. Modify training arguments in `chem_mrl/conf/base.yaml:training_args` when using Hydra-enabled training scripts.

The [PubChem 10M GenMol Fingerprint Similarity Dataset](https://huggingface.co/datasets/Derify/pubchem_10m_genmol_similarity) is available on Hugging Face Hub for training Chem-MRL models.

## Installation

**Install with pip**

```bash
pip install chem-mrl
```

**Install from source code**

```bash
pip install -e .
```

## Usage

### Hydra & Training Scripts

Hydra configuration files are in `chem_mrl/conf`. The base config defines shared arguments, while model-specific configs are located in `chem_mrl/conf/model`. Use `chem_mrl_config.yaml` or `classifier_config.yaml` to run specific models.

The `scripts` directory provides training scripts with Hydra for parameter management:

- **Train Chem-MRL model:**
  ```bash
  python scripts/train_chem_mrl.py train_dataset_path=/path/to/training.parquet val_dataset_path=/path/to/val.parquet
  ```
- **Train a linear classifier:**
  ```bash
  python scripts/train_classifier.py train_dataset_path=/path/to/training.parquet val_dataset_path=/path/to/val.parquet
  ```

### Basic Training Workflow

To train a model, initialize the configuration with dataset paths and model parameters, then pass it to `ChemMRLTrainer` for training.

```python
from sentence_transformers import SentenceTransformerTrainingArguments

from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.schemas import BaseConfig, ChemMRLConfig
from chem_mrl.trainers import ChemMRLTrainer

# Define training configuration
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
    training_args=SentenceTransformerTrainingArguments("training_output"),
    train_dataset_path="train.parquet",  # Path to training data
    val_dataset_path="val.parquet",  # Path to validation data
    test_dataset_path="test.parquet",  # Optional test dataset
    smiles_a_column_name="smiles_a",  # Column with first molecule SMILES representation
    smiles_b_column_name="smiles_b",  # Column with second molecule SMILES representation
    label_column_name="similarity",  # Similarity score between molecules
)

# Initialize trainer and start training
trainer = ChemMRLTrainer(config)
test_eval_metric = (
    trainer.train()
)  # Returns the test evaluation metric if a test dataset is provided.
# Otherwise returns the final validation eval metric
```

### Experimental

#### Train a Query Model

To train a querying model, configure the model to utilize the specialized query tokenizer.

The query tokenizer supports the following query types:

- similar: Computes SMILES similarity between two molecular structures. For retrieving similar SMILES.
- substructure: Determines the presence of a substructure within the second SMILES string.

Supported query formats for `smiles_a` column:

- `similar {smiles}`
- `substructure {smiles}`

```python
from sentence_transformers import SentenceTransformerTrainingArguments

from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.schemas import BaseConfig, ChemMRLConfig
from chem_mrl.trainers import ChemMRLTrainer

config = BaseConfig(
    model=ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        use_query_tokenizer=True,  # Train a query model
    ),
    training_args=SentenceTransformerTrainingArguments("training_output"),
    train_dataset_path="train.parquet",
    val_dataset_path="val.parquet",
    smiles_a_column_name="query",
    smiles_b_column_name="target_smiles",
    label_column_name="similarity",
)
trainer = ChemMRLTrainer(config)
```

#### Latent Attention Layer

The Latent Attention Layer model is an experimental component designed to enhance the representation learning of transformer-based models by introducing a trainable latent dictionary. This mechanism applies cross-attention between token embeddings and a set of learnable latent vectors before pooling. The output of this layer contributes to both **1D Matryoshka loss** (as the final layer output) and **2D Matryoshka loss** (by integrating into all-layer outputs). Note: initial tests suggests that when using default configuration, the latent attention layer leads to overfitting.

```python
from sentence_transformers import SentenceTransformerTrainingArguments

from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.schemas import BaseConfig, ChemMRLConfig, LatentAttentionConfig
from chem_mrl.trainers import ChemMRLTrainer

config = BaseConfig(
    model=ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        latent_attention_config=LatentAttentionConfig(
            hidden_dim=768,  # Transformer hidden size
            num_latents=512,  # Number of learnable latents
            num_cross_heads=8,  # Number of attention heads
            cross_head_dim=32,  # Dimensionality of each head
            output_normalize=True,  # Apply L2 normalization to outputs
        ),
        use_2d_matryoshka=True,
    ),
    training_args=SentenceTransformerTrainingArguments("training_output"),
    train_dataset_path="train.parquet",
    val_dataset_path="val.parquet",
)

# Train a model with latent attention
trainer = ChemMRLTrainer(config)
```

### Custom Callbacks

You can provide a list of transformers.TrainerCallback classes to execute while training.

```python
from typing import Any

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
)
from transformers import TrainerCallback, TrainerControl, TrainerState

from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.schemas import BaseConfig, ChemMRLConfig
from chem_mrl.trainers import ChemMRLTrainer


# Define a callback class for logging evaluation metrics
class EvalCallback(TrainerCallback):
    def on_evaluate(
        self,
        args: SentenceTransformerTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any],
        model: SentenceTransformer,
        **kwargs,
    ) -> None:
        """
        Event called after an evaluation phase.
        """
        pass


config = BaseConfig(
    model=ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
    ),
    training_args=SentenceTransformerTrainingArguments("training_output"),
    train_dataset_path="train.parquet",
    val_dataset_path="val.parquet",
    smiles_a_column_name="smiles_a",
    smiles_b_column_name="smiles_b",
    label_column_name="similarity",
)

# Train with callback
trainer = ChemMRLTrainer(config)
val_eval_metric = trainer.train(callbacks=[EvalCallback(...)])
```

## Classifier

This repository includes code for training a linear classifier with optional dropout regularization. The classifier categorizes substances based on SMILES and category features.

Hyperparameter tuning shows that cross-entropy loss (`softmax` option) outperforms self-adjusting dice loss in terms of accuracy, making it the preferred choice for molecular property classification.

### Usage

#### Basic Classification Training

To train a classifier, configure the model with dataset paths and column names, then initialize `ClassifierTrainer` to start training.

```python
from sentence_transformers import SentenceTransformerTrainingArguments

from chem_mrl.schemas import BaseConfig, ClassifierConfig
from chem_mrl.trainers import ClassifierTrainer

# Define classification training configuration
config = BaseConfig(
    model=ClassifierConfig(
        model_name="path/to/trained_mrl_model",  # Pretrained MRL model path
    ),
    training_args=SentenceTransformerTrainingArguments("training_output"),
    train_dataset_path="train_classification.parquet",  # Path to training dataset
    val_dataset_path="val_classification.parquet",  # Path to validation dataset
    smiles_a_column_name="smiles",  # Column containing SMILES representations of molecules
    label_column_name="label",  # Column containing classification labels
)

# Initialize and train the classifier
trainer = ClassifierTrainer(config)
trainer.train()
```

#### Training with Dice Loss

For imbalanced classification tasks, **Dice Loss** can improve performance by focusing on hard-to-classify samples. Below is a configuration using `DiceLossClassifierConfig`, which introduces additional hyperparameters.

```python
from sentence_transformers import SentenceTransformerTrainingArguments

from chem_mrl.schemas import BaseConfig, ClassifierConfig
from chem_mrl.schemas.Enums import ClassifierLossFctOption, DiceReductionOption
from chem_mrl.trainers import ClassifierTrainer

# Define classification training configuration with Dice Loss
config = BaseConfig(
    model=ClassifierConfig(
        model_name="path/to/trained_mrl_model",
        loss_func=ClassifierLossFctOption.selfadjdice,
        dice_reduction=DiceReductionOption.sum,  # Reduction method for Dice Loss (e.g., 'mean' or 'sum')
        dice_gamma=1.0,  # Smoothing factor hyperparameter
    ),
    training_args=SentenceTransformerTrainingArguments("training_output"),
    train_dataset_path="train_classification.parquet",  # Path to training dataset
    val_dataset_path="val_classification.parquet",  # Path to validation dataset
    smiles_a_column_name="smiles",
    label_column_name="label",
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
- Lee, Chankyu, et al. "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models." _arXiv [Cs.CL]_, 2025. [Link](https://arxiv.org/abs/2405.17428).
