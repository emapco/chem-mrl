# CHEM-MRL

Chem-MRL is a SMILES embedding transformer model that leverages Matryoshka Representation Learning (MRL) to generate efficient, truncatable embeddings for downstream tasks such as classification, clustering, and database querying.

The dataset (split 75%/15%/10% for train/val/test) consists of SMILES pairs and their corresponding [Morgan fingerprint](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints) (8192-bit vectors) Tanimoto similarity scores. The model employs [SentenceTransformers' (SBERT)](https://sbert.net/) [2D Matryoshka Sentence Embeddings](https://sbert.net/examples/training/matryoshka/README.html) (`Matryoshka2dLoss`) to enable truncatable embeddings with minimal accuracy loss, improving query performance in downstream applications.

Hyperparameter tuning indicates that a custom Tanimoto similarity loss function, based on CoSENTLoss, outperforms [Tanimoto similarity](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2), CoSENTLoss, [AnglELoss](https://arxiv.org/pdf/2309.12871), and cosine similarity.

## Installation

```bash
pip install chem-mrl
```

## Classifier

This repository includes code for training a linear SBERT classifier with optional dropout regularization. The classifier categorizes substances based on SMILES and category features. While demonstrated on the Isomer Design dataset, it is generalizable to any dataset containing `smiles` and `label` columns. The training scripts (see below) allow users to specify these column names.

Currently, the dataset must be in Parquet format.

Hyperparameter tuning shows that cross-entropy loss (`softmax` option) outperforms self-adjusting dice loss in terms of accuracy, making it the preferred choice for molecular property classification.

## Scripts

The `scripts` directory contains two training scripts:

- `scripts/train_chem_mrl.py` – Trains a Chem-MRL model
- `scripts/train_classifier.py` – Trains a linear classifier

### train_chem_mrl.py

For usage details, run:

```bash
python scripts/train_chem_mrl.py -h
```

Example output:

```
usage: train_chem_mrl.py [-h] --train_dataset_path TRAIN_DATASET_PATH --val_dataset_path VAL_DATASET_PATH [--test_dataset_path TEST_DATASET_PATH]
                         [--n_train_samples N_TRAIN_SAMPLES] [--n_val_samples N_VAL_SAMPLES] [--n_test_samples N_TEST_SAMPLES] [--n_dataloader_workers N_DATALOADER_WORKERS]
                         [--generate_dataset_examples_at_init] [--model_name MODEL_NAME] [--train_batch_size TRAIN_BATCH_SIZE] [--num_epochs NUM_EPOCHS] [--lr_base LR_BASE]
                         [--scheduler {warmupconstant,warmuplinear,warmupcosine,warmupcosinewithhardrestarts}] [--warmup_steps_percent WARMUP_STEPS_PERCENT] [--use_fused_adamw]
                         [--use_tf32] [--use_amp] [--seed SEED] [--model_output_path MODEL_OUTPUT_PATH] [--evaluation_steps EVALUATION_STEPS]
                         [--checkpoint_save_steps CHECKPOINT_SAVE_STEPS] [--checkpoint_save_total_limit CHECKPOINT_SAVE_TOTAL_LIMIT] [--return_eval_metric] [--use_wandb]
                         [--wandb_api_key WANDB_API_KEY] [--wandb_project_name WANDB_PROJECT_NAME] [--wandb_run_name WANDB_RUN_NAME] [--wandb_use_watch]
                         [--wandb_watch_log {gradients,parameters,all}] [--wandb_watch_log_freq WANDB_WATCH_LOG_FREQ] [--wandb_watch_log_graph]
                         [--smiles_a_column_name SMILES_A_COLUMN_NAME] [--smiles_b_column_name SMILES_B_COLUMN_NAME] [--label_column_name LABEL_COLUMN_NAME]
                         [--embedding_pooling {mean,mean_sqrt_len_tokens,weightedmean,lasttoken}] [--loss_func {tanimotosentloss,tanimotosimilarityloss,cosentloss,angleloss}]
                         [--tanimoto_similarity_loss_func {mse,l1,smooth_l1,huber,bin_cross_entropy,kldiv,cosine_embedding_loss}] [--eval_similarity_fct {cosine,tanimoto}]
                         [--eval_metric {spearman,pearson}] [--mrl_dimensions MRL_DIMENSIONS [MRL_DIMENSIONS ...]] [--dim_weights DIM_WEIGHTS [DIM_WEIGHTS ...]]
                         [--n_dims_per_step N_DIMS_PER_STEP] [--use_2d_matryoshka] [--n_layers_per_step N_LAYERS_PER_STEP] [--last_layer_weight LAST_LAYER_WEIGHT]
                         [--prior_layers_weight PRIOR_LAYERS_WEIGHT] [--kl_div_weight KL_DIV_WEIGHT] [--kl_temperature KL_TEMPERATURE]

Train SMILES-based MRL embeddings model

options:
  -h, --help            show this help message and exit
  --train_dataset_path TRAIN_DATASET_PATH
  --val_dataset_path VAL_DATASET_PATH
  --test_dataset_path TEST_DATASET_PATH
  --n_train_samples N_TRAIN_SAMPLES
                        Number of training samples to load. Uses seeded sampling if a seed is set. (default: None)
  --n_val_samples N_VAL_SAMPLES
                        Number of evaluation samples to load. Uses seeded sampling if a seed is set. (default: None)
  --n_test_samples N_TEST_SAMPLES
                        Number of testing samples to load. Uses seeded sampling if a seed is set. (default: None)
  --n_dataloader_workers N_DATALOADER_WORKERS
                        How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
  --generate_dataset_examples_at_init
                        If set, then all `sentence_transformers.InputExample` examples will be generated at at initialization. If not set, the `sentence_transformers.InputExample`
                        examples are generated on the fly by the dataloader. (default: False)
  --model_name MODEL_NAME
                        Name of the model to use. Must be either a file path or a hugging-face model name. (default: seyonec/SMILES_tokenized_PubChem_shard00_160k)
  --train_batch_size TRAIN_BATCH_SIZE
                        Training batch size (default: 32)
  --num_epochs NUM_EPOCHS
                        Number of epochs to train (default: 3)
  --lr_base LR_BASE     Base learning rate. Will be scaled by the square root of the batch size (default: 1.1190785944700813e-05)
  --scheduler {warmupconstant,warmuplinear,warmupcosine,warmupcosinewithhardrestarts}
                        Learning rate scheduler (default: warmuplinear)
  --warmup_steps_percent WARMUP_STEPS_PERCENT
                        Number of warmup steps that the scheduler will use (default: 0.0)
  --use_fused_adamw     Use cuda-optimized FusedAdamW optimizer. ~10% faster than torch.optim.AdamW (default: False)
  --use_tf32            Use TensorFloat-32 for matrix multiplication and convolutions (default: False)
  --use_amp             Use automatic mixed precision (default: False)
  --seed SEED           Omit to not set a seed during training. Used to seed the dataloader sampling and the transformer. (default: 42)
  --model_output_path MODEL_OUTPUT_PATH
                        Path to save model (default: output)
  --evaluation_steps EVALUATION_STEPS
                        Run evaluator every evaluation_steps (default: 0)
  --checkpoint_save_steps CHECKPOINT_SAVE_STEPS
                        Save checkpoint every checkpoint_save_steps (default: 0)
  --checkpoint_save_total_limit CHECKPOINT_SAVE_TOTAL_LIMIT
                        Save total limit (default: 20)
  --return_eval_metric  Return the final evaluation metric after training (default: False)
  --use_wandb           Use W&B for logging. Must be enabled for other W&B features to work. (default: False)
  --wandb_api_key WANDB_API_KEY
                        W&B API key. Can be omitted if W&B cli is installed and logged in (default: None)
  --wandb_project_name WANDB_PROJECT_NAME
  --wandb_run_name WANDB_RUN_NAME
  --wandb_use_watch     Enable W&B watch (default: False)
  --wandb_watch_log {gradients,parameters,all}
                        Specify which logs to W&B should watch (default: all)
  --wandb_watch_log_freq WANDB_WATCH_LOG_FREQ
                        How often to log (default: 1000)
  --wandb_watch_log_graph
                        Specify if graphs should be logged by W&B (default: False)
  --smiles_a_column_name SMILES_A_COLUMN_NAME
                        SMILES A column name (default: smiles_a)
  --smiles_b_column_name SMILES_B_COLUMN_NAME
                        SMILES B column name (default: smiles_b)
  --label_column_name LABEL_COLUMN_NAME
                        Label column name (default: fingerprint_similarity)
  --embedding_pooling {mean,mean_sqrt_len_tokens,weightedmean,lasttoken}
                        Pooling layer method applied to the embeddings.Pooling layer is required to generate a fixed sized SMILES embedding from a variable sized SMILES.For details
                        visit: https://sbert.net/docs/package_reference/sentence_transformer/models.html#sentence_transformers.models.Pooling (default: mean)
  --loss_func {tanimotosentloss,tanimotosimilarityloss,cosentloss,angleloss}
                        Loss function (default: tanimotosentloss)
  --tanimoto_similarity_loss_func {mse,l1,smooth_l1,huber,bin_cross_entropy,kldiv,cosine_embedding_loss}
                        Base loss function for tanimoto similarity loss function (only for tanimotosimilarityloss) (default: None)
  --eval_similarity_fct {cosine,tanimoto}
                        Similarity function to use for evaluation (default: tanimoto)
  --eval_metric {spearman,pearson}
                        Metric to use for evaluation (default: spearman)
  --mrl_dimensions MRL_DIMENSIONS [MRL_DIMENSIONS ...]
                        A list of embedding dimensions to be used for the loss function. Each value must be less than equal to the base transformer's hidden dimension. (default:
                        [768, 512, 256, 128, 64, 32, 16, 8])
  --dim_weights DIM_WEIGHTS [DIM_WEIGHTS ...]
                        A list of weights to be used for the loss function. The number of dimension weights must match that of the MRL dimensions. (default: [1, 1, 1, 1, 1, 1, 1,
                        1])
  --n_dims_per_step N_DIMS_PER_STEP
                        The number of dimensions to use per step. If -1, then all dimensions are used. If > 0, then a random sample of n_dims_per_step dimensions are used per step.
                        (default: 1)
  --use_2d_matryoshka   Use 2D Matryoshka to train over layers in addition to embedding dimensions. (default: False)
  --n_layers_per_step N_LAYERS_PER_STEP
                        The number of layers to use per step. If -1, then all layers are used. If > 0, then a random sample of n_layers_per_step layers are used per step. (only for
                        2D MRL) (default: 1)
  --last_layer_weight LAST_LAYER_WEIGHT
                        The weight to use for the loss of the final layer. Increase this to focus more on the performance when using all layers. (only for 2D MRL) (default: 1.0)
  --prior_layers_weight PRIOR_LAYERS_WEIGHT
                        The weight to use for the loss of the prior layers. Increase this to focus more on the performance when using fewer layers. (only for 2D MRL) (default: 1.0)
  --kl_div_weight KL_DIV_WEIGHT
                        The weight to use for the KL-div loss that is used to make the prior layers match that of the last layer. Increase this to focus more on the performance
                        when using fewer layers. (only for 2D MRL) (default: 1.0)
  --kl_temperature KL_TEMPERATURE
                        The temperature to use for the KL-divergence loss. If 0, then the KL-divergence loss is not used. (only for 2D MRL) (default: 0.3)
```

### train_classifier.py

For usage details, run:

```bash
$ python scripts/train_classifier.py -h
```

Example output:

```
usage: train_classifier.py [-h] --train_dataset_path TRAIN_DATASET_PATH --val_dataset_path VAL_DATASET_PATH [--test_dataset_path TEST_DATASET_PATH]
                           [--n_train_samples N_TRAIN_SAMPLES] [--n_val_samples N_VAL_SAMPLES] [--n_test_samples N_TEST_SAMPLES] [--n_dataloader_workers N_DATALOADER_WORKERS]
                           [--generate_dataset_examples_at_init] [--model_name MODEL_NAME] [--train_batch_size TRAIN_BATCH_SIZE] [--num_epochs NUM_EPOCHS] [--lr_base LR_BASE]
                           [--scheduler {warmupconstant,warmuplinear,warmupcosine,warmupcosinewithhardrestarts}] [--warmup_steps_percent WARMUP_STEPS_PERCENT] [--use_fused_adamw]
                           [--use_tf32] [--use_amp] [--seed SEED] [--model_output_path MODEL_OUTPUT_PATH] [--evaluation_steps EVALUATION_STEPS]
                           [--checkpoint_save_steps CHECKPOINT_SAVE_STEPS] [--checkpoint_save_total_limit CHECKPOINT_SAVE_TOTAL_LIMIT] [--return_eval_metric] [--use_wandb]
                           [--wandb_api_key WANDB_API_KEY] [--wandb_project_name WANDB_PROJECT_NAME] [--wandb_run_name WANDB_RUN_NAME] [--wandb_use_watch]
                           [--wandb_watch_log {gradients,parameters,all}] [--wandb_watch_log_freq WANDB_WATCH_LOG_FREQ] [--wandb_watch_log_graph]
                           [--smiles_column_name SMILES_COLUMN_NAME] [--label_column_name LABEL_COLUMN_NAME] [--eval_metric {accuracy}] [--loss_func {softmax,selfadjdice}]
                           [--classifier_hidden_dimension CLASSIFIER_HIDDEN_DIMENSION] [--dropout_p DROPOUT_P] [--freeze_model] [--dice_reduction {mean,sum}]
                           [--dice_gamma DICE_GAMMA]

Train SMILES-based classifier model

options:
  -h, --help            show this help message and exit
  --train_dataset_path TRAIN_DATASET_PATH
  --val_dataset_path VAL_DATASET_PATH
  --test_dataset_path TEST_DATASET_PATH
  --n_train_samples N_TRAIN_SAMPLES
                        Number of training samples to load. Uses seeded sampling if a seed is set. (default: None)
  --n_val_samples N_VAL_SAMPLES
                        Number of evaluation samples to load. Uses seeded sampling if a seed is set. (default: None)
  --n_test_samples N_TEST_SAMPLES
                        Number of testing samples to load. Uses seeded sampling if a seed is set. (default: None)
  --n_dataloader_workers N_DATALOADER_WORKERS
                        How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
  --generate_dataset_examples_at_init
                        If set, then all `sentence_transformers.InputExample` examples will be generated at at initialization. If not set, the `sentence_transformers.InputExample`
                        examples are generated on the fly by the dataloader. (default: False)
  --model_name MODEL_NAME
                        Name of the model to use. Must be either a file path or a hugging-face model name. (default: seyonec/SMILES_tokenized_PubChem_shard00_160k)
  --train_batch_size TRAIN_BATCH_SIZE
                        Training batch size (default: 32)
  --num_epochs NUM_EPOCHS
                        Number of epochs to train (default: 3)
  --lr_base LR_BASE     Base learning rate. Will be scaled by the square root of the batch size (default: 1.1190785944700813e-05)
  --scheduler {warmupconstant,warmuplinear,warmupcosine,warmupcosinewithhardrestarts}
                        Learning rate scheduler (default: warmuplinear)
  --warmup_steps_percent WARMUP_STEPS_PERCENT
                        Number of warmup steps that the scheduler will use (default: 0.0)
  --use_fused_adamw     Use cuda-optimized FusedAdamW optimizer. ~10% faster than torch.optim.AdamW (default: False)
  --use_tf32            Use TensorFloat-32 for matrix multiplication and convolutions (default: False)
  --use_amp             Use automatic mixed precision (default: False)
  --seed SEED           Omit to not set a seed during training. Used to seed the dataloader sampling and the transformer. (default: 42)
  --model_output_path MODEL_OUTPUT_PATH
                        Path to save model (default: output)
  --evaluation_steps EVALUATION_STEPS
                        Run evaluator every evaluation_steps (default: 0)
  --checkpoint_save_steps CHECKPOINT_SAVE_STEPS
                        Save checkpoint every checkpoint_save_steps (default: 0)
  --checkpoint_save_total_limit CHECKPOINT_SAVE_TOTAL_LIMIT
                        Save total limit (default: 20)
  --return_eval_metric  Return the final evaluation metric after training (default: False)
  --use_wandb           Use W&B for logging. Must be enabled for other W&B features to work. (default: False)
  --wandb_api_key WANDB_API_KEY
                        W&B API key. Can be omitted if W&B cli is installed and logged in (default: None)
  --wandb_project_name WANDB_PROJECT_NAME
  --wandb_run_name WANDB_RUN_NAME
  --wandb_use_watch     Enable W&B watch (default: False)
  --wandb_watch_log {gradients,parameters,all}
                        Specify which logs to W&B should watch (default: all)
  --wandb_watch_log_freq WANDB_WATCH_LOG_FREQ
                        How often to log (default: 1000)
  --wandb_watch_log_graph
                        Specify if graphs should be logged by W&B (default: False)
  --smiles_column_name SMILES_COLUMN_NAME
                        SMILES column name (default: smiles)
  --label_column_name LABEL_COLUMN_NAME
                        Label column name (default: label)
  --eval_metric {accuracy}
                        Metric to use for evaluation (default: accuracy)
  --loss_func {softmax,selfadjdice}
                        Loss function (default: softmax)
  --classifier_hidden_dimension CLASSIFIER_HIDDEN_DIMENSION
                        Classifier hidden dimension. Must be less than equal to the ChemMRL transformer's hidden dimension. Note, the base model will be truncated to this
                        dimension. (default: 768)
  --dropout_p DROPOUT_P
                        Dropout probability for linear layer regularization (default: 0.15)
  --freeze_model        Freeze internal base SMILES model (default: False)
  --dice_reduction {mean,sum}
                        Dice loss reduction. Used if loss_func=selfadjdice (default: mean)
  --dice_gamma DICE_GAMMA
                        Dice loss gamma. Used if loss_func=selfadjdice (default: 1.0)
```

## References:

- Chithrananda, Seyone, et al. "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." _arXiv [Cs.LG]_, 2020. [Link](http://arxiv.org/abs/2010.09885).
- Ahmad, Walid, et al. "ChemBERTa-2: Towards Chemical Foundation Models." _arXiv [Cs.LG]_, 2022. [Link](http://arxiv.org/abs/2209.01712).
- Kusupati, Aditya, et al. "Matryoshka Representation Learning." _arXiv [Cs.LG]_, 2022. [Link](https://arxiv.org/abs/2205.13147).
- Li, Xianming, et al. "2D Matryoshka Sentence Embeddings." _arXiv [Cs.CL]_, 2024. [Link](http://arxiv.org/abs/2402.14776).
- Bajusz, Dávid, et al. "Why is the Tanimoto Index an Appropriate Choice for Fingerprint-Based Similarity Calculations?" _J Cheminform_, 7, 20 (2015). [Link](https://doi.org/10.1186/s13321-015-0069-3).
- Li, Xiaoya, et al. "Dice Loss for Data-imbalanced NLP Tasks." _arXiv [Cs.CL]_, 2020. [Link](https://arxiv.org/abs/1911.02855)
