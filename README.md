# CHEM-MRL

[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ChemMRL%20Collection-FFD21E)](https://huggingface.co/collections/Derify/chemmrl)
![PyPI - Version](https://img.shields.io/pypi/v/chem-mrl)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/chem-mrl?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/chem-mrl)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/emapco/chem-mrl/ci.yml)
![PyPI - Status](https://img.shields.io/pypi/status/chem-mrl)

Chem-MRL is a SMILES embedding transformer model that leverages Matryoshka Representation Learning (MRL) to generate efficient, truncatable embeddings for downstream tasks such as classification, clustering, and database indexing.

Datasets should consist of SMILES pairs and their corresponding [Morgan fingerprint](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints) Tanimoto similarity scores.

Hyperparameter optimization indicates that a custom Tanimoto similarity loss function, [`TanimotoSentLoss`](https://github.com/emapco/chem-mrl/blob/main/chem_mrl/losses/TanimotoLoss.py), based on [CoSENTLoss](https://kexue.fm/archives/8847), outperforms CoSENTLoss, [AnglELoss](https://arxiv.org/pdf/2309.12871), [Tanimoto similarity](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2), and cosine similarity.

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

### Inference

ChemMRL provides [pre-trained models](https://huggingface.co/collections/Derify/chemmrl) for generating molecular embeddings from SMILES strings. The model supports various precision configurations to balance accuracy, speed, and memory usage.

#### Quick Start

```python
from chem_mrl import ChemMRL

# Load the model from the 🤗 Hub
model = ChemMRL(
    similarity_fn_name="tanimoto",  # tanimoto (default) | cosine | dot
    model_kwargs={
        "dtype": "float16",  # float32 | float16 | bfloat16
        "attn_implementation": "sdpa",  # eager | sdpa | flash_attention_2
    },
)

# Encode SMILES to embeddings
smiles = [
    "OCCCc1cc(F)cc(F)c1",
    "Fc1cc(F)cc(-n2cc[o+]n2)c1",
    "CCC(C)C(=O)C1(C(NN)C(C)C)CCCC1",
]
embeddings = model.embed(smiles)

# Calculate similarity matrix using Tanimoto similarity
similarities = model.similarity(embeddings, embeddings)
# tensor([[1.0000, 0.3875, 0.0080],
#         [0.3875, 1.0000, 0.0029],
#         [0.0080, 0.0029, 1.0000]], dtype=torch.float16)

# Calculate similarity between a molecule against a list of SMILES
query = 5 * ["CN(C)CCc1c[nH]c2cccc(OP(=O)(O)O)c12"]
docs = [
    "CN(C)CCc1c[nH]c2cccc(OP(=O)(O)Cl)c12",
    "CN(C)CCc1c[nH]c2cccc(OP(=O)(O)OP(=O)(O)O)c12",
    "CCN(C)CCc1c[nH]c2cccc(OP(=O)(O)O)c12",
    "C[N+](C)(C)CCc1c[nH]c2cccc(OP(=O)(O)O)c12",
    "CN(C)CCc1c[nH]c2cccc(OP(=O)(Cl)Cl)c12",
]
query_embeddings = model.embed(query)
docs_embeddings = model.embed(docs)
similarities = model.similarity_pairwise(query_embeddings, docs_embeddings)
# tensor([0.9697, 0.9214, 0.9751, 0.8892, 0.9067], dtype=torch.float16)
```

#### Precision Configuration

<details><summary>Click to expand the precision configuration section</summary>

Different precision and optimization settings offer trade-offs between accuracy, inference speed, and memory usage. The table below lists recommended configurations with their performance characteristics. All metrics were benchmarked with [`scripts/evaluate_precision.py`](https://github.com/emapco/chem-mrl/blob/main/scripts/evaluate_precision.py) on 131K samples (batch size = 1024), comparing speed and memory usage against the float32 baseline.

| Configuration                   | Speedup\*†    | Memory Savings\*† | Accuracy Impact     |
| ------------------------------- | ------------- | ----------------- | ------------------- |
| **bf16 (sdpa)**                 | 2.12x / 1.99x | 49.9% / 49.9%     | Minimal (~0.01%)    |
| **bf16 + torch.compile (sdpa)** | 2.55× / 2.36x | 41.9% / 41.8%     | Minimal (~0.01%)    |
| **bf16 (flash-attn)**           | 2.54× / 2.25x | 64.3% / 64.3%     | Minimal (~0.01%)    |
| **fp16 (sdpa)**                 | 2.09x / 2.00x | 49.9% / 49.9%     | Negligible (<0.01%) |
| **fp16 + torch.compile (sdpa)** | 2.56× / 2.35x | 44.7% / 44.0%     | Negligible (<0.01%) |
| **fp16 (flash-attn)**           | 2.52× / 2.25x | 64.2% / 64.1%     | Negligible (<0.01%) |

\* NVIDIA 4070 Ti Super and NVIDIA 3090 FE values respectively <br/>
† Higher is better

##### Code Examples
```python
# bfloat16 with SDPA
model = ChemMRL(
    model_kwargs={
        "dtype": "bfloat16",
        "attn_implementation": "sdpa"
    }
)
```
```python
# bfloat16 with torch.compile and SDPA
model = ChemMRL(
    model_kwargs={
        "dtype": "bfloat16",
        "attn_implementation": "sdpa"
    },
    compile_kwargs={
        "backend": "inductor",
        "dynamic": True
    }
)
```
```python
# bfloat16 with Flash Attention
model = ChemMRL(
    model_kwargs={
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2"
    }
)
```
```python
# float16 with SDPA
model = ChemMRL(
    model_kwargs={
        "dtype": "float16",
        "attn_implementation": "sdpa"
    }
)
```
```python
# float16 with torch.compile and SDPA
model = ChemMRL(
    model_kwargs={
        "dtype": "float16",
        "attn_implementation": "sdpa"
    },
    compile_kwargs={
        "backend": "inductor",
        "dynamic": True
    }
)
```
```python
# float16 with Flash Attention
model = ChemMRL(
    model_kwargs={
        "dtype": "float16",
        "attn_implementation": "flash_attention_2"
    }
)
```

</details>

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

# Use a different custom config also located in `chem_mrl/conf`
python scripts/train_chem_mrl.py --config-name=my_custom_config.yaml
```

**Configuration Options:**
- **Command line overrides:** Use `model=<type>` and parameter overrides as shown above
- **Modify base.yaml directly:** Edit the `- /model: chem_mrl` line in the defaults section to change the default model, or modify any other parameters directly
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

### MaxPoolBERT Pooling

ChemMRL supports **MaxPoolBERT**, a custom pooling strategy that combines max pooling across transformer layers with multi-head attention. This advanced pooling method can improve embedding quality by aggregating information from multiple layers instead of using only the final layer.

**Note:** MaxPoolBERT is only compatible with 1D Matryoshka models. It cannot be used when `use_2d_matryoshka=True`.

#### Pooling Strategies

MaxPoolBERT supports six pooling strategies:

- **`cls`**: Use the [CLS] token from the final layer
- **`max_cls`**: Max pool [CLS] tokens across the last k layers
- **`mha`**: Multi-head attention on the final layer sequence
- **`max_seq_mha`** (default): Max pool across last k layers, then apply multi-head attention
- **`mean_seq_mha`**: Mean pool across last k layers, then apply multi-head attention
- **`sum_seq_mha`**: Sum pool across last k layers, then apply multi-head attention

#### Enabling MaxPoolBERT via Hydra

To enable MaxPoolBERT in training scripts using Hydra configuration:

```bash
# Enable with default settings (mha strategy, 4 attention heads, last 3 layers)
python scripts/train_chem_mrl.py model.max_pool_bert.enable=true

# Customize pooling strategy and parameters
python scripts/train_chem_mrl.py \
    model.max_pool_bert.enable=true \
    model.max_pool_bert.pooling_strategy=mha \
    model.max_pool_bert.num_attention_heads=8 \
    model.max_pool_bert.last_k_layers=4
```

You can also modify the configuration directly in `chem_mrl/conf/model/chem_mrl.yaml`:

```yaml
max_pool_bert:
  enable: true
  num_attention_heads: 8
  last_k_layers: 4
  pooling_strategy: max_seq_mha
```

#### Enabling MaxPoolBERT Programmatically

To use MaxPoolBERT in your training code:

```python
from sentence_transformers import SentenceTransformerTrainingArguments

from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.schemas import BaseConfig, ChemMRLConfig, DatasetConfig, MaxPoolBERTConfig, SplitConfig
from chem_mrl.schemas.Enums import FieldTypeOption, MaxPoolBERTStrategyOption
from chem_mrl.trainers import ChemMRLTrainer

dataset_config = DatasetConfig(
    key="pubchem_10m_genmol_similarity",
    train_dataset=SplitConfig(
        name="Derify/pubchem_10m_genmol_similarity",
        split_key="train",
        label_cast_type=FieldTypeOption.float32,
        sample_size=1000,
    ),
    val_dataset=SplitConfig(
        name="Derify/pubchem_10m_genmol_similarity",
        split_key="validation",
        label_cast_type=FieldTypeOption.float32,
        sample_size=500,
    ),
    smiles_a_column_name="smiles_a",
    smiles_b_column_name="smiles_b",
    label_column_name="similarity",
)

config = BaseConfig(
    model=ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        max_pool_bert=MaxPoolBERTConfig(
            enable=True,
            pooling_strategy=MaxPoolBERTStrategyOption.max_seq_mha,
            num_attention_heads=8,
            last_k_layers=4,
        ),
    ),
    datasets=[dataset_config],
    training_args=SentenceTransformerTrainingArguments(
        "training_output",
        # bf16=True,  # Use bf16 if supported
        fp16=True,  # Use fp16 if bf16 not supported
        num_train_epochs=1,
        eval_strategy="epoch",
    ),
)

trainer = ChemMRLTrainer(config)
trainer.train()
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
        """Event called after an evaluation phase."""
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
- Behrendt, Maike, et al. "MaxPoolBERT: Enhancing BERT Classification via Layer- and Token-Wise Aggregation." _arXiv [Cs.CL]_, 2025. [Link](https://arxiv.org/abs/2505.15696).

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
