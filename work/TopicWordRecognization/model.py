from paddle import nn
from paddlenlp.transformers import ErniePretrainedModel


class ErnieNER(ErniePretrainedModel):
    def __init__(self, ernie, label_dim, dropout=None):
        super(ErnieNER, self).__init__()
        self.label_num = label_dim

        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config['hidden_size'], self.label_num)
        self.hidden = nn.Linear(self.ernie.config['hidden_size'], self.ernie.config['hidden_size'])

    def forward(self,
                words_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                history_ids=None):
        sequence_output, pooled_output = self.ernie(
            words_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = nn.functional.relu(self.hidden(self.dropout(sequence_output)))

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits