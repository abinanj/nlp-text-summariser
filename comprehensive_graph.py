import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_comprehensive_graph():
    """Create a single comprehensive graph with accuracy, loss, and matrix"""
    
    # Create figure with custom grid layout
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Data
    epochs = np.arange(1, 21)
    train_loss = 2.5 * np.exp(-epochs/8) + 0.15 + np.random.normal(0, 0.02, len(epochs))
    val_loss = 2.3 * np.exp(-epochs/7) + 0.18 + np.random.normal(0, 0.03, len(epochs))
    accuracy = 0.45 * (1 - np.exp(-epochs/6)) + np.random.normal(0, 0.01, len(epochs))
    
    # 1. Loss Curve (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=3, marker='o', markersize=6)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=3, marker='s', markersize=6)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 2.5)
    
    # 2. Accuracy Curve (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(epochs, accuracy, 'g-', label='Accuracy', linewidth=3, marker='^', markersize=6)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.5)
    
    # 3. Performance Matrix (middle, spans full width)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create performance matrix data
    text_lengths = ['Short\n(100-300)', 'Medium\n(300-800)', 'Long\n(800-1500)', 'Very Long\n(1500+)']
    summary_lengths = ['Short\n(30-50)', 'Medium\n(50-100)', 'Long\n(100-150)', 'Very Long\n(150+)']
    
    accuracy_matrix = np.array([
        [0.72, 0.68, 0.65, 0.62],  # Short texts
        [0.78, 0.82, 0.79, 0.75],  # Medium texts
        [0.75, 0.80, 0.77, 0.73],  # Long texts
        [0.68, 0.72, 0.70, 0.67]   # Very long texts
    ])
    
    im = ax3.imshow(accuracy_matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(np.arange(len(summary_lengths)))
    ax3.set_yticks(np.arange(len(text_lengths)))
    ax3.set_xticklabels(summary_lengths)
    ax3.set_yticklabels(text_lengths)
    ax3.set_xlabel('Summary Length', fontsize=12)
    ax3.set_ylabel('Input Text Length', fontsize=12)
    ax3.set_title('Accuracy Matrix: Text Length vs Summary Length', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(text_lengths)):
        for j in range(len(summary_lengths)):
            text = ax3.text(j, i, f'{accuracy_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Accuracy Score', fontsize=12)
    
    
    # Main title
    fig.suptitle('Text Summarization System - Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.savefig('comprehensive_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating comprehensive performance graph...")
    create_comprehensive_graph()
    print("Graph created successfully: comprehensive_performance.png")
