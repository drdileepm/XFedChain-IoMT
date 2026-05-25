import torch
import copy
import numpy as np

class HierarchicalFederatedServer:
    """
    Implements Hierarchical Federated Aggregation Protocol (HFAP) with DP.
    """
    def __init__(self, global_model, clip_bound=1.0, noise_multiplier=0.5):
        self.global_model = global_model
        self.clip_bound = clip_bound        # Norm clipping scale threshold C [cite: 205, 211]
        self.sigma = noise_multiplier       # Privacy noise scalar multiplier sigma [cite: 209, 211]

    def local_update_and_privatize(self, local_model, baseline_weights):
        """
        Calculates local shifts, applies norm boundaries, and injects calibrated noise[cite: 204, 205, 209].
        """
        local_delta = {}
        with torch.no_grad():
            for name, param in local_model.state_dict().items():
                if name in baseline_weights:
                    # 1. Extract structural trajectory change [cite: 204]
                    delta = baseline_weights[name] - param
                    
                    # 2. Bound sensitivity via l2-norm clipping factor [cite: 205]
                    flat_delta = delta.flatten()
                    l2_norm = torch.norm(flat_delta, p=2)
                    clip_factor = max(1.0, float(l2_norm / self.clip_bound))
                    clipped_delta = delta / clip_factor
                    
                    # 3. Add calibrated Gaussian differential privacy layer [cite: 209]
                    noise = torch.randn_like(clipped_delta) * (self.sigma * self.clip_bound)
                    privatized_delta = clipped_delta + noise
                    
                    local_delta[name] = privatized_delta
                    
        return local_delta

    def aggregate_global_model(self, verified_updates, contribution_weights):
        """
        Performs contribution-weighted parameter shifts on validated updates[cite: 216, 217].
        """
        global_state = self.global_model.state_dict()
        total_weight = sum(contribution_weights.values())
        
        with torch.no_grad():
            for name in global_state.keys():
                if name in verified_updates[0]:
                    weighted_delta_sum = torch.zeros_like(global_state[name], dtype=torch.float32)
                    for client_id, client_update in enumerate(verified_updates):
                        # Apply contribution scaling factors phi [cite: 217]
                        w = contribution_weights[client_id] / total_weight
                        weighted_delta_sum += w * client_update[name]
                    
                    # Step global weights forward using filtered aggregates
                    global_state[name] -= weighted_delta_sum
                    
        self.global_model.load_state_dict(global_state)
        return self.global_model