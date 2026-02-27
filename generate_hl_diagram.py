
import torch
import yaml
import hiddenlayer as hl
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from as_net.domain.models.as_net import ASNetConfig, DecoderConfig, EncoderConfig, SeparationModuleConfig
from as_net.infrastructure.models.as_net_torch import ASNet

def generate_diagram():
    """Generates the AS-Net architecture diagram using HiddenLayer."""
    print("--- Generating diagram with HiddenLayer ---")

    # 1. Load model configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = ASNetConfig(
        encoder=EncoderConfig(**config["model"]["encoder"]),
        separation=SeparationModuleConfig(**config["model"]["separation"]),
        decoder=DecoderConfig(**config["model"]["decoder"]),
    )
    
    # 2. Build the model
    model = ASNet(model_config)

    # 3. Create a dummy input tensor
    # Using 1 second of audio at the configured sample rate
    sample_rate = config.get("audio", {}).get("sample_rate", 22050)
    dummy_input = torch.randn(1, sample_rate)

    # 4. Create the graph
    # The `transforms` argument helps to group layers for a cleaner look
    graph = hl.build_graph(model, dummy_input, 
                           transforms=[
                               # Fold Conv, BN, ReLU layers into one block
                               hl.transforms.Fold("Conv1d > GroupNorm > PReLU > Dropout", "Conv-GN-PReLU-Drop"),
                               # Fold TCN blocks into a single conceptual block
                               hl.transforms.Fold("TCNBlock", "TCNBlock"),
                           ])

    # 5. Save the graph to a PDF
    output_filename = "as_net_architecture_hiddenlayer"
    graph.save(output_filename, format="pdf")
    
    print(f"Diagram generated successfully: {output_filename}.pdf")

if __name__ == "__main__":
    generate_diagram()
