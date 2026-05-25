from src.model import XFedChainBackbone
from src.federated import HierarchicalFederatedServer

# Initialize model trunk configurations [cite: 109]
global_net = XFedChainBackbone(num_classes=3)
hfap_orchestrator = HierarchicalFederatedServer(global_net, clip_bound=1.0, noise_multiplier=0.5)

print("XFedChain-IoMT stack successfully deployed.")