from torch import Tensor, tensor, nn
import sentence_transformers
from sentence_transformers import util, SentenceTransformer
from sentence_transformers.losses import CoSENTLoss


def pairwise_tanimoto_similarity(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes the Tanimoto similarity between two numpy arrays x and y.
    https://arxiv.org/pdf/2302.05666.pdf
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2

    Similarity based on equation (3) in 2303.5666:
    <x,y> / (||x||^2 + ||y||^2 - <x,y>) where ||a|| is the L2 norm of a.

    Note equation (3) performs better than equation (6) as a loss function metric.
    """
    if not isinstance(x, Tensor):
        x = tensor(x)

    if not isinstance(y, Tensor):
        y = tensor(y)

    dot_product = util.pairwise_dot_score(x, y)
    x_norm = util.normalize_embeddings(x)
    y_norm = util.normalize_embeddings(y)

    x_norm_squared = x_norm.pow(2).sum(dim=-1)
    y_norm_squared = y_norm.pow(2).sum(dim=-1)

    similarity = dot_product / (x_norm_squared + y_norm_squared - dot_product)

    return similarity


class TanimotoLoss(CoSENTLoss):
    def __init__(
        self,
        model: sentence_transformers.SentenceTransformer,
        scale: float = 20.0,
    ):
        """
        This class implements a variation of Tanimoto loss.

        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair.

        It computes the following loss function:

        ``loss = logsum(1+exp(s(k,l)-s(i,j))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        pairs of input pairs in the batch that match this condition. This is the same as CoSENTLoss, with the Tanimoto
        similarity function instead of the pairwise_cos_sim.

        :param model: SentenceTransformerModel
        :param scale: Output of similarity function is multiplied by scale value. Represents the inverse temperature.

        Requirements:
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Relations:
            - :class:`TanimotoLoss` is CoSENTLoss with ``pairwise_tanimoto_similarity`` as the metric, rather than ``pairwise_cos_sim``.

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+
        """
        super().__init__(
            model,
            scale,
            similarity_fct=pairwise_tanimoto_similarity,
        )


class TanimotoSimilarity(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct=util.pairwise_cos_sim,
    ):
        """
        This class implements CoSENT (Cosine Sentence) loss.
        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair.

        It computes the following loss function:

        ``loss = logsum(1+exp(s(k,l)-s(i,j))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        pairs of input pairs in the batch that match this condition.

        Anecdotal experiments show that this loss function produces a more powerful training signal than :class:`CosineSimilarityLoss`,
        resulting in faster convergence and a final model with superior performance. Consequently, CoSENTLoss may be used
        as a drop-in replacement for :class:`CosineSimilarityLoss` in any training script.

        :param model: SentenceTransformerModel
        :param similarity_fct: Function to compute the PAIRWISE similarity between embeddings. Default is ``util.pairwise_cos_sim``.
        :param scale: Output of similarity function is multiplied by scale value. Represents the inverse temperature.

        References:
            - For further details, see: https://kexue.fm/archives/8847

        Requirements:
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Relations:
            - :class:`AnglELoss` is CoSENTLoss with ``pairwise_angle_sim`` as the metric, rather than ``pairwise_cos_sim``.
            - :class:`CosineSimilarityLoss` seems to produce a weaker training signal than CoSENTLoss. In our experiments, CoSENTLoss is recommended.

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses
                from sentence_transformers.readers import InputExample

                model = SentenceTransformer('bert-base-uncased')
                train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=1.0),
                        InputExample(texts=['My third sentence', 'Unrelated sentence'], label=0.3)]

                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
                train_loss = losses.CoSENTLoss(model=model)
        """
        super(CoSENTLoss, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.scale = scale

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        scores = self.similarity_fct(embeddings[0], embeddings[1])
        scores = scores * self.scale
        scores = scores[:, None] - scores[None, :]

        # label matrix indicating which pairs are relevant
        labels = labels[:, None] < labels[None, :]
        labels = labels.float()

        # mask out irrelevant pairs so they are negligible after exp()
        scores = scores - (1 - labels) * 1e12

        # append a zero as e^0 = 1
        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        loss = torch.logsumexp(scores, dim=0)

        return loss

    def get_config_dict(self):
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}
