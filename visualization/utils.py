from typing import List
import matplotlib.pyplot as plt


def setup_plot_style():
    """Setup common plot styling."""
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10

def get_color_palette(num_colors: int) -> List[str]:
    """Get colorblind-friendly color palette."""
    base_colors = [
        '#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6',
        '#1abc9c', '#e67e22', '#34495e', '#7f8c8d', '#16a085'
    ]
    colors = base_colors * (num_colors // len(base_colors) + 1)
    return colors[:num_colors]

