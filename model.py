import torch
from torch import nn
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from scipy.ndimage import distance_transform_edt


class SimpleSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_labels, dropout_prob=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_labels, kernel_size=1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, patch_embeddings):
        B, N, C = patch_embeddings.shape
        H = W = int(N ** 0.5)
        if H * W != N:
            raise ValueError(f"Number of patches {N} does not form a square grid.")
        x = patch_embeddings.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.dropout(x)
        return self.conv(x)


class Dinov2ForSkySegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.dinov2 = Dinov2Model(config)
        self.classifier = SimpleSegmentationHead(config.hidden_size, num_labels=1)
        self.config.num_labels = 1

    def forward(self, pixel_values, labels=None):
        # Extract patch embeddings without computing gradients
        with torch.no_grad():
            outputs = self.dinov2(pixel_values,
                                  output_hidden_states=False,
                                  output_attentions=False)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        logits = self.classifier(patch_embeddings)
        logits = nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        loss = None
        if labels is not None:
            # Prepare distance-based weights to penalize predictions near true boundaries
            labels_np = labels.cpu().numpy()
            weight_list = []
            for i in range(labels_np.shape[0]):
                # distance to nearest background for sky pixels
                dist = distance_transform_edt(labels_np[i, 0])
                # inverse distance weight, add 1 to avoid div by zero
                w = 1.0 / (dist + 1.0)**1.3
                weight_list.append(w)
            # Stack and convert to tensor
            weight_tensor = torch.tensor(
                weight_list, device=labels.device, dtype=logits.dtype
            ).unsqueeze(1)
            # Compute weighted BCE loss
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(), weight=weight_tensor
            )

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
