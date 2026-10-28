import torch
from torch import nn
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class LinearRoberta(RobertaPreTrainedModel):
    """Identical CLS linear head for all methods; no nonlinear hidden head."""
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.method = 'full'
        self.post_init()

    def configure(self, method):
        if method not in ('M1', 'M2', 'full'):
            raise ValueError(method)
        self.method = method
        for p in self.roberta.parameters():
            p.requires_grad = method == 'full'
        if method == 'M2':
            if len(self.roberta.encoder.layer) < 2:
                raise ValueError('M2 requires at least two encoder layers')
            for layer in self.roberta.encoder.layer[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Frozen representations are deterministic. M2 dropout stays active
        # only in the last two trainable encoder layers during training.
        if mode and self.method in ('M1', 'M2'):
            self.roberta.eval()
            if self.method == 'M2':
                for layer in self.roberta.encoder.layer[-2:]:
                    layer.train(True)
        return self

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids, return_dict=True)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        loss = None if labels is None else nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
