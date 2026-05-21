import torch
from onnx2torch import convert

model = convert('/home/kushalb/leela-nla/lc0.onnx')

# Dictionary to store the captured activations
activations = {}

def get_activation(name):
    """Returns a forward hook that stores the module's output."""
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Define the layer you want to hook (e.g., the end of the first encoder)
layer_to_hook = 'encoder0/mha/out/skip'

# Register the hook on the specified layer
hooked = False
for name, module in model.named_modules():
    if name == layer_to_hook:
        module.register_forward_hook(get_activation(layer_to_hook))
        print(f"Hook registered on layer: {layer_to_hook}")
        hooked = True
        break

if not hooked:
    print(f"Could not find layer: {layer_to_hook}")

# Let's run a forward pass with dummy data to trigger the hook.
# From ONNX graph, the input is named '/input/planes' with shape (batch, 112, 8, 8).
print("Running a forward pass with a dummy input (shape: 1, 112, 8, 8)...")
dummy_input = torch.randn(1, 112, 8, 8)

# Forward pass
output = model(dummy_input)

# Verify the activation was captured
if layer_to_hook in activations:
    captured_tensor = activations[layer_to_hook]
    print(f"\nSUCCESS! Captured activation for '{layer_to_hook}'.")
    print(f"Activation shape: {captured_tensor.shape}")
    print(f"Activation tensor summary (mean, std): {captured_tensor.mean().item():.4f}, {captured_tensor.std().item():.4f}")
else:
    print(f"\nFailed to capture activation for '{layer_to_hook}'.")
