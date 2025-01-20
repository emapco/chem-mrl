# CHEM-MRL

CHEM-MRL is a SMILES embedding transformer model that uses Matryoshka Representation Learning (MRL) to compute efficient, truncatable embeddings for various downstream tasks, including classification, clustering, and database querying. The dataset (75%/15%/10% train/val/test split) consists of SMILES pairs and their corresponding [Morgan fingerprints](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints) (8192-bit vectors) with Tanimoto similarity. The model utilizes the [Sentence Bert (SBERT)](https://sbert.net/) `2D-MRL` implementation to enable truncatable embeddings with minimal accuracy loss, enhancing querying performance for downstream tasks.

Hyperparameter tuning indicates that a custom Tanimoto similarity loss function based on CoSENTLoss, performs better than the [Tanimoto](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2) distance, CoSENTLoss, [AnglELoss](https://arxiv.org/pdf/2309.12871), and cosine [distance (Eq 1.)](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3#Equ1).

The `visualization` directory contains embeddings of the [Isomer Design](https://isomerdesign.com/pihkal/search) SMILES dataset, with various embedding sizes and models (functional vs. non-functional Morgan fingerprints). The `.tsv` files can be visualized using [TensorFlow Projector](https://projector.tensorflow.org/).

## Classifier

The repository includes code for training a linear SBERT classifier with dropout, designed to categorize substances based on SMILES and category features. While it is specifically demonstrated on the Isomer Design dataset, the model is highly generalizable and can be applied to any dataset containing `smiles` and `category` features. For more information on how to prepare and load your data, refer to the `load_data` function in `classifier/load_data.py`.

During training, the model was evaluated using both self-adjusting dice loss and cross-entropy loss. Hyperparameter tuning determined that cross-entropy loss outperformed self-adjusting dice loss in terms of model accuracy, making it the preferred choice for training on molecular property datasets. This classifier is well-suited for classification tasks across a wide range of chemical datasets, allowing for efficient training and generalization.

## References:
- Chithrananda, Seyone, et al. "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." *arXiv [Cs.LG]*, 2020. [Link](http://arxiv.org/abs/2010.09885).
- Ahmad, Walid, et al. "ChemBERTa-2: Towards Chemical Foundation Models." *arXiv [Cs.LG]*, 2022. [Link](http://arxiv.org/abs/2209.01712).
- Li, Xianming, et al. "2D Matryoshka Sentence Embeddings." *arXiv [Cs.CL]*, 2024. [Link](http://arxiv.org/abs/2402.14776).
- Bajusz, D., Rácz, A., & Héberger, K. "Why is the Tanimoto Index an Appropriate Choice for Fingerprint-Based Similarity Calculations?" *J Cheminform*, 7, 20 (2015). [Link](https://doi.org/10.1186/s13321-015-0069-3).
- Li, Xiaoya, et al. "Dice Loss for Data-imbalanced NLP Tasks." *arXiv [Cs.CL]*, 2020. [Link](https://arxiv.org/abs/1911.02855)
