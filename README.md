# CHEM-MRL

Chem-MRL is a SMILES embedding transformer model that leverages Matryoshka Representation Learning (MRL) to generate efficient, truncatable embeddings for downstream tasks such as classification, clustering, and database querying.

The dataset (split 75%/15%/10% for train/val/test) consists of SMILES pairs and their corresponding [Morgan fingerprint](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints) (8192-bit vectors) Tanimoto similarity scores. The model employs [SentenceTransformers' (SBERT)](https://sbert.net/) [2D Matryoshka Sentence Embeddings](https://sbert.net/examples/training/matryoshka/README.html) (`Matryoshka2dLoss`) to enable truncatable embeddings with minimal accuracy loss, improving query performance in downstream applications.

Hyperparameter tuning indicates that a custom Tanimoto similarity loss function, based on CoSENTLoss, outperforms [Tanimoto similarity](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2), CoSENTLoss, [AnglELoss](https://arxiv.org/pdf/2309.12871), and cosine similarity.

## Installation

```bash
pip install chem-mrl
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
from chem_mrl.schemas import ChemMRLConfig
from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.trainers import ChemMRLTrainer

# Define training configuration
config = BaseConfig(
    model=ChemMRLConfig(
        model_name=BASE_MODEL_NAME,  # Predefined model name - Can be a any transformer model name or path that is compatible with sentence-transformers
        smiles_a_column_name="smiles_a",  # Column with first molecule SMILES representation
        smiles_b_column_name="smiles_b",  # Column with second molecule SMILES representation
        label_column_name="similarity",  # Similarity score between molecules
        n_dims_per_step=3,  # Model-specific hyperparameter
        use_2d_matryoshka=True,  # Enable 2d MRL
        # Additional parameters specific to 2D MRL models
        n_layers_per_step=2,
        kl_div_weight=0.7,  # Weight for KL divergence regularization
        kl_temperature=0.5,  # Temperature parameter for KL loss
    ),
    train_dataset_path="train.parquet",  # Path to training data
    val_dataset_path="val.parquet",  # Path to validation data
    test_dataset_path="test.parquet",  # Optional test dataset
)

# Initialize trainer and start training
trainer = ChemMRLTrainer(config)
test_eval_metric = (
    trainer.train()
)  # Returns evaluation metric (if test dataset exists) otherwise returns the final validation eval metric
```

### Custom Evaluation Callbacks

You can provide a callback function that is executed every `evaluation_steps` steps, allowing custom logic such as logging, early stopping, or model checkpointing.

```python
from chem_mrl.schemas import Chem2dMRLConfig
from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.trainers import ChemMRLTrainer


# Define a callback function for logging evaluation metrics
def eval_callback(score: float, epoch: int, steps: int):
    print(f"Step {steps}, Epoch {epoch}: Evaluation Score = {score}")


config = BaseConfig(
    model=ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        smiles_a_column_name="smiles_a",
        smiles_b_column_name="smiles_b",
        label_column_name="similarity",
    ),
    train_dataset_path="train.parquet",
    val_dataset_path="val.parquet",
)

# Train with callback
trainer = ChemMRLTrainer(config)
val_eval_metric = trainer.train(
    eval_callback=eval_callback
)  # Callback executed every `evaluation_steps`
```

### W&B Integration

This library includes a `WandBTrainerExecutor` class for seamless Weights & Biases (W&B) integration. It handles authentication, initialization, and logging at the frequency specified by `evaluation_steps`. This setup ensures seamless logging and experiment tracking, allowing for better visualization and monitoring of model performance.

```python
from chem_mrl.schemas import Chem2dMRLConfig, ChemMRLConfig
from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.trainers import ChemMRLTrainer, WandBTrainerExecutor

# Define W&B configuration for experiment tracking
wandb_config = WandbConfig(
    project_name="chem_mrl_test",  # W&B project name
    run_name="test",  # Name for the experiment run
    use_watch=True,  # Enables model watching for tracking gradients
    watch_log="all",  # Logs all model parameters and gradients
    watch_log_freq=1000,  # Logging frequency
    watch_log_graph=True,  # Logs model computation graph
)

# Configure training with W&B integration
config = BaseConfig(
    model=ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        smiles_a_column_name="smiles_a",
        smiles_b_column_name="smiles_b",
        label_column_name="similarity",
    ),
    train_dataset_path="train.parquet",
    val_dataset_path="val.parquet",
    evaluation_steps=1000,
    use_wandb=True,  # Enables W&B logging
    wandb_config=wandb_config,
)

# Initialize trainer and W&B executor
trainer = ChemMRLTrainer(config)
executor = WandBTrainerExecutor(trainer)
executor.execute()  # Handles training and W&B logging
```

## Classifier

This repository includes code for training a linear classifier with optional dropout regularization. The classifier categorizes substances based on SMILES and category features. While demonstrated on the Isomer Design dataset, it is generalizable to any dataset containing `smiles` and `label` columns. The training scripts (see below) allow users to specify these column names.

Currently, the dataset must be in Parquet format.

Hyperparameter tuning shows that cross-entropy loss (`softmax` option) outperforms self-adjusting dice loss in terms of accuracy, making it the preferred choice for molecular property classification.

## Usage

### Basic Classification Training

To train a classifier, configure the model with dataset paths and column names, then initialize `ClassifierTrainer` to start training.

```python
from chem_mrl.schemas import ClassifierConfig
from chem_mrl.trainers import ClassifierTrainer

# Define classification training configuration
config = BaseConfig(
    model=ClassifierConfig(
        model_name="path/to/trained_mrl_model",  # Pretrained MRL model path
        smiles_column_name="smiles",  # Column containing SMILES representations of molecules
        label_column_name="label",  # Column containing classification labels
    ),
    train_dataset_path="train_classification.parquet",  # Path to training dataset
    val_dataset_path="val_classification.parquet",  # Path to validation dataset
)

# Initialize and train the classifier
trainer = ClassifierTrainer(config)
trainer.train()
```

### Training with Dice Loss

For imbalanced classification tasks, **Dice Loss** can improve performance by focusing on hard-to-classify samples. Below is a configuration using `DiceLossClassifierConfig`, which introduces additional hyperparameters.

```python
from chem_mrl.schemas import DiceLossClassifierConfig
from chem_mrl.trainers import ClassifierTrainer
from chem_mrl.constants import BASE_MODEL_NAME

# Define classification training configuration with Dice Loss
config = BaseConfig(
    model=ClassifierConfig(
        model_name="path/to/trained_mrl_model",
        smiles_column_name="smiles",
        label_column_name="label",
        dice_reduction="sum",  # Reduction method for Dice Loss (e.g., 'mean' or 'sum')
        dice_gamma=1.0,  # Dice loss hyperparameter
    ),
    train_dataset_path="train_classification.parquet",  # Path to training dataset
    val_dataset_path="val_classification.parquet",  # Path to validation dataset
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
