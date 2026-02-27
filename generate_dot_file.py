import os
import sys

def generate_dot_file():
    """Generates a .dot file describing the AS-Net architecture."""
    print("[DEBUG] Starting generate_dot_file function...")
    
    # Using triple quotes to avoid issues with escaping quotes and newlines
    dot_string = '''
    digraph ASNet {
    // Graph Attributes
    graph [bgcolor="transparent", splines="spline", fontsize="24", ranksep="1.2"];
    node [fontsize="14", fontname="Helvetica"];
    edge [fontsize="12"];

    // --- Inputs ---
    input_signal [label="Input\nMixed Audio (1D)", shape=cylinder, style=filled, fillcolor="#7f8c8d", fontcolor="white"];
    ground_truth [label="Input\nClean Audio (Ground Truth)", shape=cylinder, style=filled, fillcolor="#7f8c8d", fontcolor="white"];

    // --- AS-Net Model Cluster ---
    subgraph cluster_model {
        label = "AS-Net Model";
        style = "filled";
        color = "#f7f7f7";
        node [shape=box, style=filled];

        // --- Nodes ---
        encoder [label="Conv1D Encoder\n(k=16, s=8, ch_in=1, ch_out=128)", fillcolor="#4285F4", fontcolor="white"];
        encoded_features [label="Encoded Features\nShape: (B, 128, T/8)", shape=note, fillcolor="#34A853", fontcolor="white"];
        
        separation [label="TCN Blocks\n(8 blocks, k=3, dilation 1→128, dropout=0.35)", fillcolor="#4285F4", fontcolor="white"];
        separated_features [label="Separated Features\nShape: (B, 128, T/8)", shape=note, fillcolor="#34A853", fontcolor="white"];
        
        mask_estimation [label="Mask Estimation\n(Conv1D, k=1, Sigmoid)", fillcolor="#4285F4", fontcolor="white"];
        masks [label="2 Masks (Bio+Noise)", shape=note, fillcolor="#34A853", fontcolor="white"];
        
        masked_mult [label="Element-wise Mult", fillcolor="#4285F4", fontcolor="white"];
        decoder [label="ConvTranspose1D\n(k=16, s=8, ch_in=128, ch_out=1)", fillcolor="#4285F4", fontcolor="white"];

        // --- Layout Ranks (for 2-column grid) ---
        { rank = same; encoder; encoded_features; }
        { rank = same; separation; separated_features; }
        { rank = same; mask_estimation; masks; }
        { rank = same; masked_mult; decoder; }

        // --- Connections ---
        encoded_features -> separation [lhead=cluster_model];
        separated_features -> mask_estimation;
        masks -> masked_mult;
        masked_mult -> decoder;
        encoder -> encoded_features [style=invis];
        separation -> separated_features [style=invis];
        mask_estimation -> masks [style=invis];
        masked_mult -> decoder [style=invis];
    }

    // --- Outputs ---
    output_bio [label="Separated Bio-acoustics", shape=cylinder, style=filled, fillcolor="#7f8c8d", fontcolor="white"];
    output_noise [label="Separated Noise", shape=cylinder, style=filled, fillcolor="#7f8c8d", fontcolor="white"];

    // --- Loss Cluster ---
    subgraph cluster_loss {
        label = "Training Objective";
        style = "filled";
        color = "#f7f7f7";
        loss_function [label="Permutation Invariant Training\nLoss = -SI-SDR(s, ŝ)", shape=plaintext];
    }

    // --- Final Connections ---
    input_signal -> encoder;
    encoded_features -> masked_mult [style=dashed, minlen=2];
    decoder -> output_bio;
    decoder -> output_noise;
    output_bio -> loss_function [minlen=2];
    ground_truth -> loss_function [minlen=2];
}
    '''

    print("[DEBUG] dot_string variable created.")

    output_filename = "architecture.dot"
    try:
        print(f"[DEBUG] Trying to write to {os.path.abspath(output_filename)}...")
        with open(output_filename, "w") as f:
            f.write(dot_string)
        print(f"[SUCCESS] DOT file generated successfully: {output_filename}")
    except (IOError, OSError) as e:
        print(f"[ERROR] Failed to write file: {e}", file=sys.stderr)

if __name__ == "__main__":
    generate_dot_file()