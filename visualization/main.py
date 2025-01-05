from typing import Dict
import matplotlib.pyplot as plt
from .utils import setup_plot_style
from .plots import (plot_fitness_evolution,
                    plot_population_diversity,
                    )
from .gantt import create_gantt_chart


def create_visualization_dashboard(history: Dict) -> plt.Figure:
    """Create comprehensive visualization dashboard."""
    setup_plot_style()

    fig = plt.figure(figsize=(15, 12))

    # Create subplots
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax4 = plt.subplot(2, 2, 3)

    # Generate plots
    plot_fitness_evolution(ax1, history)
    plot_population_diversity(ax2, history)
    create_gantt_chart(ax4, history['best_solutions'][-1].schedule,
                       history['best_solutions'][-1].problem.num_machines)

    plt.tight_layout(pad=3.0)
    return fig


def save_individual_plots(history: Dict, output_dir: str = '.'):
    """Save individual plots to separate files."""
    setup_plot_style()

    # Fitness Evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_fitness_evolution(ax, history)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fitness_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Population Diversity
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_population_diversity(ax, history)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/population_diversity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Gantt Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    create_gantt_chart(ax, history['best_solutions'][-1].schedule,
                       history['best_solutions'][-1].problem.num_machines)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gantt_chart.png', dpi=300, bbox_inches='tight')
    plt.close()