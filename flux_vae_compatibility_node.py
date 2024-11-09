# flux_vae_compatibility_node.py

import torch
import comfyUI
from comfyUI.gui import FileInput, Dropdown, Button, Label

class FluxVAECompatibilityNode(comfyUI.Node):
    def __init__(self):
        super().__init__()
        
        # Default VAE path (user will adjust in GUI)
        self.vae_model_path = ""
        
        # GUI Elements
        self.add_gui_elements()
        
        # Placeholder for loaded VAE
        self.vae = None

    def add_gui_elements(self):
        # Label for the node
        self.add(Label("Flux VAE Compatibility Node"))
        
        # File Input for selecting VAE model
        self.add(FileInput("Select VAE Model", callback=self.set_vae_path))
        
        # Dropdown for selecting channel adaptation type
        self.add(Dropdown("Expected Channels", ["4", "16"], default="4", callback=self.set_expected_channels))
        
        # Load Button for confirming VAE load
        self.add(Button("Load VAE", callback=self.load_vae))
        
        # Information label for status updates
        self.status_label = Label("")
        self.add(self.status_label)

    def set_vae_path(self, path):
        self.vae_model_path = path

    def set_expected_channels(self, channels):
        self.expected_channels = int(channels)

    def load_vae(self):
        try:
            self.vae = torch.load(self.vae_model_path)
            self.status_label.set_text("VAE loaded successfully.")
        except Exception as e:
            self.status_label.set_text(f"Failed to load VAE: {e}")

    def preprocess_input(self, input_tensor):
        # Adjust input tensor to match expected channels if needed
        if input_tensor.shape[1] != self.expected_channels:
            input_tensor = input_tensor[:, :self.expected_channels, :, :]
        return input_tensor

    def forward(self, input_tensor):
        if self.vae:
            input_tensor = self.preprocess_input(input_tensor)
            return self.vae(input_tensor)
        else:
            self.status_label.set_text("VAE not loaded. Please load a VAE model.")

# Register the node within ComfyUI
if __name__ == "__main__":
    comfyUI.register_node(FluxVAECompatibilityNode, "Flux VAE Compatibility")
