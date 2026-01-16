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

    # 3. Create the Grad-CAM object
    grad_cam = GradCAM1D(model, target_layer)

    # 4. Create a dummy input (or load real data)
    # Shape: (Batch=1, Channels=37, Length=1000)
    # input_signal = torch.randn(1, 37, 1000, requires_grad=True)

    # 5. Run Grad-CAM
    heatmap = grad_cam(input_signal)

    # 6. Plotting
    # imshow the input_signal and overlay the heatmap
    signal_data = input_signal.detach().numpy().squeeze()
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
    plt.savefig("test.png")

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

    def __call__(self, x, target_class=1):
        """
        Args:
            x: Input tensor of shape (1, n_channels, length)
        """
        # 1. Forward Pass
        output = self.model(x)

        # In your model, output is sigmoid [batch, 1]
        # We backpropagate through the logit (or sigmoid result)
        self.model.zero_grad()
        output.backward()

        # Weight the channels by the gradients
        # gradients shape: [batch, channels, length]
        weights = torch.mean(self.gradients, dim=2, keepdim=True)

        # Calculate Grad-CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to keep only positive influences
        cam = F.relu(cam)

        # Resize to match original input length
        cam = F.interpolate(cam, size=x.shape[2], mode="linear", align_corners=False)

        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Return flattened numpy array
        return cam.detach().cpu().numpy()[0, 0, :]

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
