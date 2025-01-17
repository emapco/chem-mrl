from torch import nn
from sentence_transformers import SentenceTransformer


class SentenceTransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        freeze_transformer=False,
        enable_dropout=True,
    ):
        super(SentenceTransformerClassifier, self).__init__()
        self.transformer_model = SentenceTransformer(model_name)

        if not isinstance(self.transformer_model, SentenceTransformer):
            raise TypeError("model_name must be a valid SentenceTransformer model")

        # Freeze transformer layers if specified
        if freeze_transformer:
            for p in self.transformer_model.parameters():
                p.requires_grad = False

        self.hidden_size = self.sentence_transformer.get_sentence_embedding_dimension()
        assert isinstance(self.hidden_size, int), "last layer dimension is None"

        self.num_labels = num_labels
        assert isinstance(self.num_labels, int), "num_labels must be an integer"

        self.enable_dropout = enable_dropout
        assert isinstance(self.enable_dropout, bool), "enable_dropout must be a boolean"

        self.fc = nn.Linear(self.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, sentences):
        """
        Args:
            x: text to encode

        Returns:
            Tensor: logits of the classification layer (output of the classifier layer)
        """
        sentences = self.transformer_model.encode(
            sentences, convert_to_numpy=False
        ).unsqueeze(0)

        if self.enable_dropout:
            fc_layer_logits = self.fc(self.dropout(sentences))
        else:
            fc_layer_logits = self.fc(sentences)

        return fc_layer_logits
