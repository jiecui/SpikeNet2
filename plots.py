# plots for SpikeNet2

# ==========================================================================
# Imports libraries
# ==========================================================================
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sleeplib.Resnet_15.model import ResNet


# ==========================================================================
# Defne functions
# ==========================================================================
def find_last_conv_layer(module):
    """
    Helper to automatically find the last nn.Conv1d layer in your Net1D model.
    Grad-CAM usually works best on the very last convolutional layer.
    """
    last_conv = None
    for name, sub_mod in module.named_modules():
        if isinstance(sub_mod, torch.nn.Conv1d):
            last_conv = sub_mod
    return last_conv


def run_visualization(model: ResNet, input_signal: torch.Tensor) -> None:
    # 1. Initialize your model
    # (In practice, you would load from a checkpoint)
    # model = ResNet.load_from_checkpoint("path/to/checkpoint.ckpt", ...)

    # 2. Find the target layer
    # Your ResNet wraps 'Net1D' in 'self.model'
    target_layer = find_last_conv_layer(model.model)

    if target_layer is None:
        print("Error: No Conv1d layer found.")
        return

    print(f"Applying Grad-CAM on layer: {target_layer}")

    # 3. Create the Grad-CAM object
    grad_cam = GradCAM1D(model, target_layer)

    # 4. Create a dummy input (or load real data)
    # Shape: (Batch=1, Channels=37, Length=1000)
    # input_signal = torch.randn(1, 37, 1000, requires_grad=True)

    # 5. Run Grad-CAM
    heatmap = grad_cam(input_signal)

    # 6. Plotting
    # We plot the first channel of the signal against the heatmap
    signal_data = input_signal.detach().numpy()[0, 0, :]  # Channel 0

    plt.figure(figsize=(12, 5))

    # Plot Signal
    plt.plot(signal_data, label="Input Signal (Lead 0)", color="black", alpha=0.5)

    # Plot Heatmap (Overlay)
    # We map the heatmap to the range of the signal for better visualization
    # or simply plot it on a secondary axis
    ax2 = plt.gca().twinx()
    ax2.plot(heatmap, color="red", label="Grad-CAM", linewidth=2)
    ax2.fill_between(np.arange(len(heatmap)), heatmap, color="red", alpha=0.2)
    ax2.set_ylabel("Importance", color="red")

    plt.title("Grad-CAM: Feature Importance")
    plt.show()

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

        # Register hooks
        # These hooks save the activations during forward pass
        # and gradients during backward pass
        self.handle_fwd = self.target_layer.register_forward_hook(self.save_activation)
        self.handle_bwd = self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is usually a tuple, we want the first element
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.handle_fwd.remove()
        self.handle_bwd.remove()

    def __call__(self, x):
        """
        Args:
            x: Input tensor of shape (1, n_channels, length)
        """
        # 1. Forward Pass
        self.model.eval()
        self.model.zero_grad()

        # We need the input to require grad to ensure backprop chain is valid
        # though strictly we only need weights grad, being explicit helps debugging
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Ensure batch dimension

        # Get model prediction
        logits = self.model(x)

        # 2. Select Target Score
        # Your model has n_classes=1 (Binary/Regression).
        # The output is shape (1, 1) or (1,).
        # We simply backpropagate the scalar output itself.
        # If logits is (Batch, 1), we select the 0th element.
        score = logits[0] if logits.dim() > 0 else logits

        # 3. Backward Pass
        score.backward()

        # 4. Generate Heatmap
        # gradients shape: (1, Filters, Length)
        # activations shape: (1, Filters, Length)

        gradients = self.gradients
        activations = self.activations

        # Global Average Pooling of gradients (over time dimension 2)
        # This gives us the "importance" (alpha) of each feature map
        weights = torch.mean(gradients, dim=2, keepdim=True)

        # Weighted combination of feature maps
        # (1, Filters, Length) * (1, Filters, 1)
        cam = torch.sum(weights * activations, dim=1)

        # Apply ReLU (we are only interested in features that have a positive influence)
        cam = F.relu(cam)

        # Normalize to 0-1 range for visualization
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)

        # 5. Upsample to original input size
        # We use 1D interpolation (linear)
        # Input cam is (1, Length_feature), we need (1, Length_original)
        target_length = x.shape[2]

        # Interpolate expects (Batch, Channels, Length)
        cam = cam.unsqueeze(1)
        cam = F.interpolate(cam, size=target_length, mode="linear", align_corners=False)

        # Return flattened numpy array
        return cam.squeeze().detach().cpu().numpy()
