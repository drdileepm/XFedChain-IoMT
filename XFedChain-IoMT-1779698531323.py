import torch
import torch.nn.functional as F
import numpy as np

class GradCAMPlusPlus:
    """
    Generates class-discriminative spatial saliency attention maps using higher-order gradients[cite: 307, 308].
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register extraction hooks for structural backpropagation tracking
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Target logit score Y^c extraction [cite: 309]
        target_score = output[0][target_class]
        target_score.backward()
        
        # Fetch feature tensor elements [cite: 309]
        grads = self.gradients[0]
        activations = self.activations[0]
        
        # Calculate localized importance maps alpha using partial derivatives [cite: 308, 310]
        grads_power_2 = grads.pow(2)
        grads_power_3 = grads.pow(3)
        
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)
        # Higher-order spatial gradient alignment weight system tracking [cite: 315, 317]
        denominator = 2 * grads_power_2 + sum_activations * grads_power_3
        denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        
        aij = grads_power_2 / denominator
        weights = torch.clamp(F.relu(grads), min=0) * aij
        alpha = weights.sum(dim=(1, 2), keepdim=True)
        
        # Rectified linear combination projection [cite: 320, 321]
        cam = (alpha * activations).sum(dim=0)
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam_np = cam.cpu().detach().numpy()
        cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np) + 1e-8)
        return cam_np