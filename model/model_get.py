import torch

from .GAT import GAT_Model
from .GCN import GCN_Model
from .GIN import GIN_Model
from .QWTA_base import QWTA_base
from .QWTA_GR import QWTA_GR
from .Transformer import Transformer_Model

model_dict = {
    "QWTA_base": QWTA_base,
    "QWTA_GR": QWTA_GR,
    "GAT": GAT_Model,
    "GCN": GCN_Model,
    "Transformer": Transformer_Model,
    "GIN": GIN_Model,
    "QWTA_AD": QWTA_GR,  # Model name in historical GR configs.
}


def get_model(args, num_node_features, edge_index, num_nodes, device):
    model_class = model_dict.get(args.model)
    if model_class is None:
        raise ValueError(f"Unknown model {args.model!r}. Choose from {', '.join(model_dict)}.")
    if model_class is QWTA_GR and args.irregular_time_steps < 2:
        raise ValueError("Historical QWTA-GR requires at least two retained observations.")

    model = model_class(
        in_features=num_node_features,
        hidden_dim=64,
        out_steps=args.pre_time_range,
        n_vertex=num_nodes,
        edge_index=edge_index,
        num_nodes=num_nodes,
        args=args,
    ).to(device)
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"Loaded checkpoint: {checkpoint_path}")
    return model
