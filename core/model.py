import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class CORE(nn.Module):
    """
    Correlation-Guided Representation (CORE).

    Jointly encodes text and one dedicated token per label. Computes cosine
    correlations between text-token and label-token representations, applies
    a shared local convolution over each label's correlation sequence, max
    pools over labels, normalizes attention over text positions, and predicts
    labels from the correlation-guided text representation.
    """
    def __init__(self, pretrained_model="bert-base-uncased", num_labels=103,
                 window_size=10, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_labels = num_labels

        # Shared convolution kernel across all label correlation sequences.
        self.local_conv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=window_size,
            padding=0, bias=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def resize_token_embeddings(self, new_size):
        self.encoder.resize_token_embeddings(new_size)

    def forward(self, input_ids, attention_mask, text_lengths, labels=None):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoded.last_hidden_state
        batch_size, _, hidden_size = hidden.shape
        device = hidden.device
        n_labels = self.num_labels

        # Text begins at sequence position 1, immediately after [CLS].
        max_text_len = int(text_lengths.max().item())
        text_positions = torch.arange(max_text_len, device=device).unsqueeze(0) + 1
        text_index = text_positions.unsqueeze(-1).expand(batch_size, max_text_len, hidden_size)
        Hx = torch.gather(hidden, dim=1, index=text_index)

        # Label tokens begin after [CLS] + text + [SEP], separately for each row.
        label_offsets = text_lengths.unsqueeze(1) + 2
        label_positions = label_offsets + torch.arange(n_labels, device=device).unsqueeze(0)
        label_index = label_positions.unsqueeze(-1).expand(batch_size, n_labels, hidden_size)
        Hy = torch.gather(hidden, dim=1, index=label_index)

        # L2-normalized text-label correlation matrix G: [B, text_len, labels].
        Hx_norm = F.normalize(Hx, p=2, dim=-1, eps=1e-12)
        Hy_norm = F.normalize(Hy, p=2, dim=-1, eps=1e-12)
        G = torch.bmm(Hx_norm, Hy_norm.transpose(1, 2))

        # Apply local convolution along text positions independently to each label.
        # Conv uses asymmetric padding for even window sizes so output length == input length.
        corr = G.transpose(1, 2).reshape(batch_size * n_labels, 1, max_text_len)
        k = self.local_conv.kernel_size[0]
        left = (k - 1) // 2
        right = k // 2
        corr = F.pad(corr, (left, right))
        local = F.relu(self.local_conv(corr)).squeeze(1)
        local = local.reshape(batch_size, n_labels, max_text_len)

        # Max over label dimension; ignore padded text positions.
        attention_logits = local.max(dim=1).values
        valid_text_mask = (
            torch.arange(max_text_len, device=device).unsqueeze(0)
            < text_lengths.unsqueeze(1)
        )
        attention_logits = attention_logits.masked_fill(~valid_text_mask, -1e4)
        alpha = torch.softmax(attention_logits, dim=1)
        alpha = torch.tanh(alpha)

        context = torch.bmm(alpha.unsqueeze(1), Hx).squeeze(1)
        logits = self.classifier(self.dropout(context))

        output = {"logits": logits, "attention": alpha}
        if labels is not None:
            output["loss"] = F.binary_cross_entropy_with_logits(logits, labels)
        return output
