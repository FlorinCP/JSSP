from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

from models import JobShopChromosome


def plot_fitness_evolution(history: Dict) -> plt.Figure:
    """Plot fitness evolution over generations."""

    plt.style.use('default')
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    generations = range(len(history['best_fitness']))

    # Plot main curves
    ax.plot(generations, history['best_fitness'],
            label='Best Fitness', color='#2ecc71', linewidth=2)
    ax.plot(generations, history['worst_fitness'],
            label='Worst Fitness', color='#e74c3c', linewidth=2)

    # Find and mark best and worst points
    best_gen = np.argmin(history['best_fitness'])
    best_value = history['best_fitness'][best_gen]
    worst_gen = np.argmax(history['worst_fitness'])
    worst_value = history['worst_fitness'][worst_gen]

    # Add markers for best and worst points
    ax.scatter(best_gen, best_value, color='#2ecc71', s=100, zorder=5)
    ax.scatter(worst_gen, worst_value, color='#e74c3c', s=100, zorder=5)

    # Add annotations
    ax.annotate(f'Best: {best_value:.2f}\nGeneration: {best_gen}',
                (best_gen, best_value),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='#2ecc71', alpha=0.3))

    ax.annotate(f'Worst: {worst_value:.2f}\nGeneration: {worst_gen}',
                (worst_gen, worst_value),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='#e74c3c', alpha=0.3))

    plt.title('Fitness Evolution Over Generations', fontsize=14, pad=20)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness (Total Time)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    return fig


# def plot_population_diversity(history: Dict) -> plt.Figure:
#     """Plot population diversity over generations."""
#     plt.style.use('default')
#     fig = plt.figure(figsize=(12, 8))
#     ax = plt.gca()
#     generations = range(len(history['diversity']))
#
#     # Plot diversity curve
#     ax.plot(generations, history['diversity'],
#             color='#9b59b6', linewidth=2, label='Population Diversity')
#
#     # Calculate statistics
#     max_div = max(history['diversity'])
#     min_div = min(history['diversity'])
#     avg_div = np.mean(history['diversity'])
#     final_div = history['diversity'][-1]
#
#     # Add statistics box
#     stats_text = (f'Maximum Diversity: {max_div:.2f}\n'
#                   f'Average Diversity: {avg_div:.2f}\n'
#                   f'Minimum Diversity: {min_div:.2f}\n'
#                   f'Final Diversity: {final_div:.2f}')
#
#     plt.text(0.02, 0.98, stats_text,
#              transform=ax.transAxes,
#              bbox=dict(facecolor='white', alpha=0.8, edgecolor='#9b59b6'),
#              verticalalignment='top',
#              fontsize=10)
#
#     plt.title('Population Diversity Over Generations', fontsize=14, pad=20)
#     plt.xlabel('Generation', fontsize=12)
#     plt.ylabel('Diversity Score', fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(loc='upper right', fontsize=10)
#     plt.tight_layout()
#
#     return fig

def plot_population_diversity(history: Dict) -> plt.Figure:
    """Plot population diversity over generations with detailed analysis."""
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    generations = range(len(history['diversity']))
    diversity_values = history['diversity']

    # Plot main diversity curve
    ax.plot(generations, diversity_values,
            color='#9b59b6', linewidth=2, label='Population Diversity')

    # Add rolling average for trend visualization
    window_size = min(10, len(diversity_values))
    rolling_avg = np.convolve(diversity_values, np.ones(window_size) / window_size, mode='valid')
    rolling_gens = generations[window_size - 1:]
    ax.plot(rolling_gens, rolling_avg,
            color='#2ecc71', linewidth=2, linestyle='--',
            label=f'{window_size}-Generation Moving Average')

    # Calculate comprehensive statistics
    max_div = max(diversity_values)
    min_div = min(diversity_values)
    avg_div = np.mean(diversity_values)
    final_div = diversity_values[-1]

    # Calculate rate of diversity change
    div_changes = np.diff(diversity_values)
    avg_change_rate = np.mean(div_changes)

    # Find significant points
    max_div_gen = np.argmax(diversity_values)
    min_div_gen = np.argmin(diversity_values)

    # Calculate diversity stability
    stability = np.std(diversity_values)

    # Add marker points for maximum and minimum diversity
    ax.scatter(max_div_gen, max_div, color='#2ecc71', s=100, zorder=5, label='Maximum Diversity')
    ax.scatter(min_div_gen, min_div, color='#e74c3c', s=100, zorder=5, label='Minimum Diversity')

    # Add comprehensive statistics box
    stats_text = (
        f'Diversity Statistics:\n'
        f'Maximum: {max_div:.2f} (Gen {max_div_gen})\n'
        f'Minimum: {min_div:.2f} (Gen {min_div_gen})\n'
        f'Average: {avg_div:.2f}\n'
        f'Final: {final_div:.2f}\n'
        f'Stability (σ): {stability:.2f}\n'
        f'Avg Rate of Change: {avg_change_rate:.3f}/gen'
    )

    plt.text(0.02, 0.02, stats_text,
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#9b59b6'),
             verticalalignment='top',
             fontsize=10)

    # Complete plot configuration
    plt.title('Population Diversity Analysis Over Generations', fontsize=14, pad=20)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Diversity Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='center right', fontsize=10, bbox_to_anchor=(1.15, 0.5))
    plt.tight_layout()

    return fig

def plot_schedule(chromosome: JobShopChromosome) -> plt.Figure:
    """Create Gantt chart of the schedule."""

    schedule = chromosome.decode_to_schedule()
    num_machines = chromosome.problem.num_machines

    plt.style.use('default')
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Calculate makespan by finding the latest end time of any operation
    makespan = max(details['end'] for details in schedule.values())

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6',
              '#1abc9c', '#e67e22', '#34495e', '#7f8c8d', '#16a085']

    while len(colors) < num_machines:
        colors.extend(colors)
    colors = colors[:num_machines]

    job_positions = {}

    sorted_ops = sorted(schedule.items(), key=lambda x: (x[1]['machine'], x[1]['start']))

    for (job_id, op_idx), details in sorted_ops:
        start = details['start']
        duration = details['end'] - start
        machine = details['machine']

        # Create bar
        ax.barh(y=machine, width=duration, left=start,
                color=colors[job_id % len(colors)], alpha=0.8,
                label=f'Job {job_id}' if job_id not in job_positions else "")

        # Store job position and add operation details
        job_positions[job_id] = True
        ax.text(start + duration / 2, machine, f'J{job_id}-Op{op_idx}\n({duration})',
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.title(f'Best Schedule (Total Makespan: {makespan})', fontsize=14, pad=20)
    plt.xlabel('Time Units', fontsize=12)
    plt.ylabel('Machine', fontsize=12)
    plt.yticks(range(num_machines), [f'M{i}' for i in range(num_machines)])
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    plt.xlim(0, makespan)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = plt.legend(by_label.values(), by_label.keys(),
                        title='Jobs',
                        loc='center left',
                        bbox_to_anchor=(1, 0.5),
                        frameon=True,
                        fancybox=True,
                        shadow=True)
    legend.get_title().set_fontweight('bold')

    plt.tight_layout()
    return fig
