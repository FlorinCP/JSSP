from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


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


def plot_population_diversity(history: Dict) -> plt.Figure:
    """Plot population diversity over generations."""
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    generations = range(len(history['diversity']))

    # Plot diversity curve
    ax.plot(generations, history['diversity'],
            color='#9b59b6', linewidth=2, label='Population Diversity')

    # Calculate statistics
    max_div = max(history['diversity'])
    min_div = min(history['diversity'])
    avg_div = np.mean(history['diversity'])
    final_div = history['diversity'][-1]

    # Add statistics box
    stats_text = (f'Maximum Diversity: {max_div:.2f}\n'
                  f'Average Diversity: {avg_div:.2f}\n'
                  f'Minimum Diversity: {min_div:.2f}\n'
                  f'Final Diversity: {final_div:.2f}')

    plt.text(0.02, 0.98, stats_text,
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#9b59b6'),
             verticalalignment='top',
             fontsize=10)

    plt.title('Population Diversity Over Generations', fontsize=14, pad=20)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Diversity Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    return fig


def plot_schedule(schedule: Dict, num_machines: int) -> plt.Figure:
    """Create Gantt chart of the schedule."""
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Use a colorblind-friendly palette
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6',
              '#1abc9c', '#e67e22', '#34495e', '#7f8c8d', '#16a085']
    while len(colors) < num_machines:
        colors.extend(colors)
    colors = colors[:num_machines]

    # Calculate makespan
    makespan = max(details['end'] for details in schedule.values())

    # Track job positions
    job_positions = {}

    for (job_id, op_idx), details in sorted(schedule.items(),
                                            key=lambda x: (x[1]['machine'], x[1]['start'])):
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

    # Improve legend
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