# CHEM-MRL

CHEM-MRL is a SMILES embedding transformer model that uses Matryoshka Representation Learning (MRL) to compute efficient, truncatable embeddings for various downstream tasks, including classification, clustering, and database querying. The dataset consists of SMILES pairs and their corresponding [Morgan fingerprints](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints) (8196-bit vectors) with Tanimoto similarity. The model utilizes the SBERT "2D-MRL" (preprint) implementation to enable truncatable embeddings with minimal accuracy loss, enhancing querying performance for downstream tasks.

Hyperparameter tuning indicates that a custom Tanimoto similarity loss function based on CoSENTLoss, performs better than the [Tanimoto](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2) distance, CoSENTLoss, [AnglELoss](https://arxiv.org/pdf/2309.12871), and cosine [distance (Eq 1.)](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3).

The `visualization` directory contains embeddings of the [Isomer Design](https://isomerdesign.com/pihkal/search) SMILES dataset, with various embedding sizes and models (functional vs. non-functional Morgan fingerprints). The `.tsv` files can be visualized using [TensorFlow Projector](https://projector.tensorflow.org/).

### References:
- Chithrananda, Seyone, et al. "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." *arXiv [Cs.LG]*, 2020. [Link](http://arxiv.org/abs/2010.09885).
- Ahmad, Walid, et al. "ChemBERTa-2: Towards Chemical Foundation Models." *arXiv [Cs.LG]*, 2022. [Link](http://arxiv.org/abs/2209.01712).
- Li, Xianming, et al. "2D Matryoshka Sentence Embeddings." *arXiv [Cs.CL]*, 2024. [Link](http://arxiv.org/abs/2402.14776).
- Bajusz, D., Rácz, A., & Héberger, K. "Why is the Tanimoto Index an Appropriate Choice for Fingerprint-Based Similarity Calculations?" *J Cheminform*, 7, 20 (2015). [Link](https://doi.org/10.1186/s13321-015-0069-3).
