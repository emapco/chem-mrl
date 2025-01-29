# CHEM-MRL

CHEM-MRL is a SMILES embedding transformer model that leverages Matryoshka Representation Learning (MRL) to generate efficient, truncatable embeddings for downstream tasks such as classification, clustering, and database querying.

The dataset (split 75%/15%/10% for train/val/test) consists of SMILES pairs and their corresponding [Morgan fingerprint](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints) (8192-bit vectors) Tanimoto similarity scores. The model employs [SentenceTransformers' (SBERT)](https://sbert.net/) [2D Matryoshka Sentence Embeddings](https://sbert.net/examples/training/matryoshka/README.html) (`Matryoshka2dLoss`) to enable truncatable embeddings with minimal accuracy loss, improving query performance in downstream applications.

Hyperparameter tuning indicates that a custom Tanimoto similarity loss function, based on CoSENTLoss, outperforms [Tanimoto similarity](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2), CoSENTLoss, [AnglELoss](https://arxiv.org/pdf/2309.12871), and cosine similarity.

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
```bash
usage: train_chem_mrl.py [-h] --train_dataset_path TRAIN_DATASET_PATH --val_dataset_path VAL_DATASET_PATH [--test_dataset_path TEST_DATASET_PATH]
                         [--num_train_samples NUM_TRAIN_SAMPLES] [--num_val_samples NUM_VAL_SAMPLES] [--num_test_samples NUM_TEST_SAMPLES] [--model_name MODEL_NAME]
                         [--train_batch_size TRAIN_BATCH_SIZE] [--num_epochs NUM_EPOCHS] [--lr_base LR_BASE]
                         [--scheduler {warmupconstant,warmuplinear,warmupcosine,warmupcosinewithhardrestarts}] [--warmup_steps_percent WARMUP_STEPS_PERCENT] [--use_fused_adamw]
                         [--use_tf32] [--use_amp] [--seed SEED] [--model_output_path MODEL_OUTPUT_PATH] [--evaluation_steps EVALUATION_STEPS]
                         [--checkpoint_save_steps CHECKPOINT_SAVE_STEPS] [--checkpoint_save_total_limit CHECKPOINT_SAVE_TOTAL_LIMIT] [--use_wandb]
                         [--wandb_api_key WANDB_API_KEY] [--wandb_project_name WANDB_PROJECT_NAME] [--wandb_run_name WANDB_RUN_NAME] [--wandb_use_watch]
                         [--wandb_watch_log {gradients,parameters,all}] [--wandb_watch_log_freq WANDB_WATCH_LOG_FREQ] [--wandb_watch_log_graph]
                         [--smiles_a_column_name SMILES_A_COLUMN_NAME] [--smiles_b_column_name SMILES_B_COLUMN_NAME] [--label_column_name LABEL_COLUMN_NAME]
                         [--loss_func {tanimotosentloss,tanimotosimilarityloss,cosentloss}]
                         [--tanimoto_similarity_loss_func {mse,l1,smooth_l1,huber,bin_cross_entropy,kldiv,cosine_embedding_loss}] [--eval_similarity_fct {cosine,tanimoto}]
                         [--eval_metric {spearman,pearson}] [--first_dim_weight FIRST_DIM_WEIGHT] [--second_dim_weight SECOND_DIM_WEIGHT] [--third_dim_weight THIRD_DIM_WEIGHT]
                         [--fourth_dim_weight FOURTH_DIM_WEIGHT] [--fifth_dim_weight FIFTH_DIM_WEIGHT] [--sixth_dim_weight SIXTH_DIM_WEIGHT] [--use_2d_matryoshka]
                         [--last_layer_weight LAST_LAYER_WEIGHT] [--prior_layers_weight PRIOR_LAYERS_WEIGHT]

Train SMILES-based MRL embeddings model

options:
  -h, --help            show this help message and exit
  --train_dataset_path TRAIN_DATASET_PATH
  --val_dataset_path VAL_DATASET_PATH
  --test_dataset_path TEST_DATASET_PATH
  --num_train_samples NUM_TRAIN_SAMPLES
                        Number of training samples to load. Uses seeded sampling if a seed is set. (default: None)
  --num_val_samples NUM_VAL_SAMPLES
                        Number of evaluation samples to load. Uses seeded sampling if a seed is set. (default: None)
  --num_test_samples NUM_TEST_SAMPLES
                        Number of testing samples to load. Uses seeded sampling if a seed is set. (default: None)
  --model_name MODEL_NAME
                        Name of the model to use. Either file path or a hugging-face model name. (default: seyonec/ChemBERTa-zinc-base-v1)
  --train_batch_size TRAIN_BATCH_SIZE
                        Training batch size (default: 32)
  --num_epochs NUM_EPOCHS
                        Number of epochs to train (default: 3)
  --lr_base LR_BASE     Base learning rate. Will be scaled by the batch size (default: 1.1190785944700813e-05)
  --scheduler {warmupconstant,warmuplinear,warmupcosine,warmupcosinewithhardrestarts}
                        Learning rate scheduler (default: warmuplinear)
  --warmup_steps_percent WARMUP_STEPS_PERCENT
                        Number of warmup steps that the scheduler will use (default: 0.0)
  --use_fused_adamw     Use cuda-optimized FusedAdamW optimizer. ~10% faster than torch.optim.AdamW (default: False)
  --use_tf32            Use TensorFloat-32 for matrix multiplication and convolutions (default: False)
  --use_amp             Use automatic mixed precision (default: False)
  --seed SEED           Omit to not set a seed during training. Seed dataloader sampling and transformers. (default: 42)
  --model_output_path MODEL_OUTPUT_PATH
                        Path to save model (default: output)
  --evaluation_steps EVALUATION_STEPS
                        Run evaluator every evaluation_steps (default: 0)
  --checkpoint_save_steps CHECKPOINT_SAVE_STEPS
                        Save checkpoint every checkpoint_save_steps (default: 0)
  --checkpoint_save_total_limit CHECKPOINT_SAVE_TOTAL_LIMIT
                        Save total limit (default: 20)
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
  --loss_func {tanimotosentloss,tanimotosimilarityloss,cosentloss}
                        Loss function (default: tanimotosentloss)
  --tanimoto_similarity_loss_func {mse,l1,smooth_l1,huber,bin_cross_entropy,kldiv,cosine_embedding_loss}
                        Base loss function for tanimoto similarity loss function (default: None)
  --eval_similarity_fct {cosine,tanimoto}
                        Similarity metric to use for evaluation (default: tanimoto)
  --eval_metric {spearman,pearson}
                        Metric to use for evaluation (default: spearman)
  --first_dim_weight FIRST_DIM_WEIGHT
  --second_dim_weight SECOND_DIM_WEIGHT
  --third_dim_weight THIRD_DIM_WEIGHT
  --fourth_dim_weight FOURTH_DIM_WEIGHT
  --fifth_dim_weight FIFTH_DIM_WEIGHT
  --sixth_dim_weight SIXTH_DIM_WEIGHT
  --use_2d_matryoshka   Use 2D Matryoshka (default: False)
  --last_layer_weight LAST_LAYER_WEIGHT
                        Last layer weight used by 2D Matryoshka for computing the loss (default: 1.8708220063487997)
  --prior_layers_weight PRIOR_LAYERS_WEIGHT
                        Prior layers weight used by 2D Matryoshka for computing compute the loss (default: 1.4598249321447245)
```

### train_classifier.py
For usage details, run:

```bash
$ python scripts/train_classifier.py -h
```
Example output:
```bash
usage: train_classifier.py [-h] --train_dataset_path TRAIN_DATASET_PATH --val_dataset_path VAL_DATASET_PATH [--test_dataset_path TEST_DATASET_PATH]
                           [--num_train_samples NUM_TRAIN_SAMPLES] [--num_val_samples NUM_VAL_SAMPLES] [--num_test_samples NUM_TEST_SAMPLES] [--model_name MODEL_NAME]
                           [--train_batch_size TRAIN_BATCH_SIZE] [--num_epochs NUM_EPOCHS] [--lr_base LR_BASE]
                           [--scheduler {warmupconstant,warmuplinear,warmupcosine,warmupcosinewithhardrestarts}] [--warmup_steps_percent WARMUP_STEPS_PERCENT]
                           [--use_fused_adamw] [--use_tf32] [--use_amp] [--seed SEED] [--model_output_path MODEL_OUTPUT_PATH] [--evaluation_steps EVALUATION_STEPS]
                           [--checkpoint_save_steps CHECKPOINT_SAVE_STEPS] [--checkpoint_save_total_limit CHECKPOINT_SAVE_TOTAL_LIMIT] [--use_wandb]
                           [--wandb_api_key WANDB_API_KEY] [--wandb_project_name WANDB_PROJECT_NAME] [--wandb_run_name WANDB_RUN_NAME] [--wandb_use_watch]
                           [--wandb_watch_log {gradients,parameters,all}] [--wandb_watch_log_freq WANDB_WATCH_LOG_FREQ] [--wandb_watch_log_graph]
                           [--smiles_column_name SMILES_COLUMN_NAME] [--label_column_name LABEL_COLUMN_NAME] [--eval_metric {accuracy}] [--loss_func {softmax,selfadjdice}]
                           [--classifier_hidden_dimension {768,512,256,128,64,32}] [--dropout_p DROPOUT_P] [--freeze_model] [--dice_reduction {mean,sum}]
                           [--dice_gamma DICE_GAMMA]

Train SMILES-based classifier model

options:
  -h, --help            show this help message and exit
  --train_dataset_path TRAIN_DATASET_PATH
  --val_dataset_path VAL_DATASET_PATH
  --test_dataset_path TEST_DATASET_PATH
  --num_train_samples NUM_TRAIN_SAMPLES
                        Number of training samples to load. Uses seeded sampling if a seed is set. (default: None)
  --num_val_samples NUM_VAL_SAMPLES
                        Number of evaluation samples to load. Uses seeded sampling if a seed is set. (default: None)
  --num_test_samples NUM_TEST_SAMPLES
                        Number of testing samples to load. Uses seeded sampling if a seed is set. (default: None)
  --model_name MODEL_NAME
                        Name of the model to use. Either file path or a hugging-face model name. (default: seyonec/ChemBERTa-zinc-base-v1)
  --train_batch_size TRAIN_BATCH_SIZE
                        Training batch size (default: 32)
  --num_epochs NUM_EPOCHS
                        Number of epochs to train (default: 3)
  --lr_base LR_BASE     Base learning rate. Will be scaled by the batch size (default: 1.1190785944700813e-05)
  --scheduler {warmupconstant,warmuplinear,warmupcosine,warmupcosinewithhardrestarts}
                        Learning rate scheduler (default: warmuplinear)
  --warmup_steps_percent WARMUP_STEPS_PERCENT
                        Number of warmup steps that the scheduler will use (default: 0.0)
  --use_fused_adamw     Use cuda-optimized FusedAdamW optimizer. ~10% faster than torch.optim.AdamW (default: False)
  --use_tf32            Use TensorFloat-32 for matrix multiplication and convolutions (default: False)
  --use_amp             Use automatic mixed precision (default: False)
  --seed SEED           Omit to not set a seed during training. Seed dataloader sampling and transformers. (default: 42)
  --model_output_path MODEL_OUTPUT_PATH
                        Path to save model (default: output)
  --evaluation_steps EVALUATION_STEPS
                        Run evaluator every evaluation_steps (default: 0)
  --checkpoint_save_steps CHECKPOINT_SAVE_STEPS
                        Save checkpoint every checkpoint_save_steps (default: 0)
  --checkpoint_save_total_limit CHECKPOINT_SAVE_TOTAL_LIMIT
                        Save total limit (default: 20)
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
  --classifier_hidden_dimension {768,512,256,128,64,32}
                        Classifier hidden dimension. The base SMILES model will be truncated to this dimension (default: 768)
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
