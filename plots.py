# plots for SpikeNet2

# 2026 Richard J. Cui. Modified: Fri 01/16/2026 03:19:02.190139 PM
# $Revision: 0.1 $  $Date: Fri 01/16/2026 03:19:02.190139 PM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

# ==========================================================================
# Imports libraries
# ==========================================================================
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sleeplib.Resnet_15.model import ResNet


# ==========================================================================
# Define functions
# ==========================================================================
def find_last_conv_layer(module):
    """
    Helper to automatically find the last nn.Conv1d layer in your Net1D model.
    Grad-CAM usually works best on the very last convolutional layer.
    """
    return module.stage_list[-1].block_list[-1].conv3


def run_visualization(model: ResNet, input_signal: torch.Tensor) -> None:
    # 1. Initialize your model
    # (In practice, you would load from a checkpoint)
    # model = ResNet.load_from_checkpoint("path/to/checkpoint.ckpt", ...)
    model.eval()

    # 2. Find the target layer
    # Your ResNet wraps 'Net1D' in 'self.model'
    target_layer = find_last_conv_layer(model.model)

    if target_layer is None:
        print("Error: No Conv1d layer found.")
        return

    print(f"Applying Grad-CAM on layer: {target_layer}")

    with torch.enable_grad():
        # 3. Create the Grad-CAM object
        grad_cam = GradCAM1D(model, target_layer)
        # 4. Run Grad-CAM
        heatmap = grad_cam(input_signal)

    # 4. Plotting
    # imshow the input_signal and overlay the heatmap
    signal_data = input_signal.cpu().detach().numpy().squeeze()
    # normalize signal for better visualization
    signal_data = (signal_data - np.min(signal_data)) / (
        np.max(signal_data) - np.min(signal_data)
    )

    plt.figure(figsize=(12, 5))

    # Plot Signal
    plt.imshow(signal_data, aspect="auto", cmap="gray", alpha=0.7)

    # Plot Heatmap (Overlay)
    # We map the heatmap to the range of the signal for better visualization
    # or simply plot it on a secondary axis
    ax2 = plt.gca().twinx()
    ax2.plot(heatmap, color="red", label="Grad-CAM", linewidth=2)
    ax2.fill_between(np.arange(len(heatmap)), heatmap, color="red", alpha=0.2)
    ax2.set_ylabel("Importance", color="red")

    plt.title("Grad-CAM: Feature Importance")
    plt.show()
    # plt.savefig("test.png")

    # Clean up hooks
    grad_cam.remove_hooks()


# ==========================================================================
# Define classes
# ==========================================================================
class GradCAM1D:
    def __init__(self, model, target_layer):
        """
        Args:
            model: The PyTorch Lightning module (ResNet).
            target_layer: The specific layer to compute gradients on (usually the last Conv1d).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def __call__(self, x):
        """
        Args:
            x: Input tensor of shape (1, n_channels, length)
        """
        # Move to device and enable gradients
        x.requires_grad = True

        # Forward pass
        self.model.zero_grad()
        output = self.model(x)  # shape: [1, 1], binary classification

        # Backprop from scalar
        score = output[0]
        score.backward()

        # gradients: [batch, channels, length]
        # Validate that hooks captured data
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients or activations")
        # weights = torch.mean(self.gradients, dim=2, keepdim=True) # global average pooling [batch, channels, 1]

        # Use per-time gradients directly:
        weights = self.gradients  # Keep all [batch, channels, length] information

        # temporal Grad-CAM
        cam = torch.sum(
            weights * self.activations, dim=1, keepdim=True
        )  # [batch, 1, length]
        cam = F.relu(cam)

        # Resize to input length
        cam = F.interpolate(cam, size=x.shape[2], mode="linear", align_corners=False)

        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.cpu().detach().numpy()[0, 0, :]

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
