"""
architecture_diagram.py
=========================
Generates a publication-quality diagram of the HaRAG pipeline architecture.
Uses matplotlib to create a flow-based visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_rag_architecture(save_path: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Define common box properties
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', lw=1.5)
    arrow_props = dict(arrowstyle='->', lw=2, color='darkgray')

    # 1. Input Node
    ax.text(50, 95, "User Query", ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.5', facecolor='#E3F2FD'))
    ax.annotate("", xy=(50, 88), xytext=(50, 92), arrowprops=arrow_props)

    # 2. Retriever
    ax.text(50, 85, "Hybrid Retriever\n(BM25 + Dense)", ha='center', va='center', bbox=box_props)
    ax.annotate("", xy=(50, 78), xytext=(50, 82), arrowprops=arrow_props)

    # 3. Context & Generator
    ax.text(50, 75, "Relevant Context", ha='center', va='center', style='italic')
    ax.annotate("", xy=(50, 72), xytext=(50, 74), arrowprops=arrow_props)
    ax.text(50, 68, "LLM Generator\n(FLAN-T5 / Llama)", ha='center', va='center', bbox=box_props)
    ax.annotate("", xy=(50, 61), xytext=(50, 65), arrowprops=arrow_props)

    # 4. Answer Node
    ax.text(50, 58, "Generated Answer", ha='center', va='center', weight='bold')

    # Split to parallel components
    ax.annotate("", xy=(25, 48), xytext=(45, 56), arrowprops=arrow_props)
    ax.annotate("", xy=(50, 48), xytext=(50, 56), arrowprops=arrow_props)
    ax.annotate("", xy=(75, 48), xytext=(55, 56), arrowprops=arrow_props)

    # 5. Analysis Layer
    ax.text(25, 45, "Evidence\nHighlighter", ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#FFF9C4'))
    ax.text(50, 45, "Contradiction\nVerifier", ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#F8BBD0'))
    ax.text(75, 45, "Hallucination\nDetector", ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#C8E6C9'))

    # Re-converge to VCS
    ax.annotate("", xy=(50, 32), xytext=(25, 42), arrowprops=arrow_props)
    ax.annotate("", xy=(50, 32), xytext=(50, 42), arrowprops=arrow_props)
    ax.annotate("", xy=(50, 32), xytext=(75, 42), arrowprops=arrow_props)

    # 6. VCS Meta-Model
    ax.text(50, 28, "VCS Meta-Model\n(Learned Weights)", ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#BBDEFB', lw=2))
    ax.annotate("", xy=(50, 20), xytext=(50, 24), arrowprops=arrow_props)

    # 7. Calibration
    ax.text(50, 17, "Temperature Scaling\n(Calibration)", ha='center', va='center', 
            bbox=dict(boxstyle='Sawtooth', facecolor='#FFCCBC'))
    ax.annotate("", xy=(50, 9), xytext=(50, 14), arrowprops=arrow_props)

    # 8. Final Output
    ax.text(50, 5, "FINAL ANSWER\n+ Confidence (VCS)", ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='#FFFDE7', edgecolor='gold', lw=3))

    plt.title("HaRAG: Hallucination-Aware RAG Pipeline Architecture", pad=20, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram saved to {save_path}")

if __name__ == "__main__":
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_rag_architecture(str(out_dir / "architecture_diagram.png"))
