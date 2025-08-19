from datasets import load_dataset

from chem_mrl import ChemMRL
from chem_mrl.evaluation import EmbeddingSimilarityEvaluator
from chem_mrl.schemas.Enums import ChemMrlEvalMetricOption, EvalSimilarityFctOption

chem_mrl = ChemMRL("Derify/ChemMRL-beta")
dataset = load_dataset("Derify/pubchem_10m_genmol_similarity", split="test")

evaluator = EmbeddingSimilarityEvaluator(
    dataset["smiles_a"],
    dataset["smiles_b"],
    dataset["similarity"],
    batch_size=1024,
    main_similarity=EvalSimilarityFctOption.tanimoto,
    metric=ChemMrlEvalMetricOption.spearman,
    name="pubchem_10m_genmol_similarity",
    show_progress_bar=True,
    write_csv=True,
    precision="float32",
)

chem_mrl.backbone.evaluate(evaluator, output_path="evaluation_results")
