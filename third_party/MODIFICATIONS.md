# Modifications to Third-Party Code

This document details modifications made to third-party software components used in this project. All modifications are made under the Apache License 2.0 and comply with the original license terms.

## Sentence Transformers - [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)

Modifications under Apache License 2.0:

- **chem_mrl/evaluation/LabelAccuracyEvaluator.py**

  - Extracted CSV writing logic into a reusable function.

- **chem_mrl/evaluation/EmbeddingSimilarityEvaluator.py**
  - Extracted CSV writing logic into a reusable function.
  - Added Tanimoto similarity while removing dot product, Euclidean, and Manhattan similarity functions.
  - Reduced memory usage by computing a single similarity score instead of multiple.
  - Modified class to call custom similarity functions, which downcast float precision to further reduce memory usage.

For details on the original code, refer to the [Sentence Transformers repository](https://github.com/UKPLab/sentence-transformers).
