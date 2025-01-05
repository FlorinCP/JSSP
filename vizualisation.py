from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


def plot_genetic_algorithm_stats(history: Dict):
    """Create comprehensive visualization of GA performance."""
    # Use a built-in style that's guaranteed to exist
    plt.style.use('default')

    # Create figure with better spacing
    fig = plt.figure(figsize=(15, 12))
    plt.rcParams['figure.facecolor'] = 'white'

    # Plot 1: Fitness Evolution
    ax1 = plt.subplot(2, 2, 1)
    generations = range(len(history['best_fitness']))

    ax1.plot(generations, history['best_fitness'],
             label='Best', color='#2ecc71', linewidth=2)
    ax1.plot(generations, history['worst_fitness'],
             label='Worst', color='#e74c3c', linewidth=2)
    ax1.plot(generations, history['avg_fitness'],
             label='Average', color='#3498db', linewidth=2)

    ax1.set_title('Fitness Evolution Over Generations', fontsize=12, pad=15)
    ax1.set_xlabel('Generation', fontsize=10)
    ax1.set_ylabel('Fitness (Total Time)', fontsize=10)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Population Diversity
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(generations, history['diversity'],
             color='#9b59b6', linewidth=2)
    ax2.fill_between(generations, history['diversity'],
                     alpha=0.2, color='#9b59b6')

    ax2.set_title('Population Diversity', fontsize=12, pad=15)
    ax2.set_xlabel('Generation', fontsize=10)
    ax2.set_ylabel('Diversity Score', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Improvement Rate
    ax3 = plt.subplot(2, 2, 3)
    improvements = np.diff(history['best_fitness'])
    ax3.plot(generations[1:], improvements,
             color='#f1c40f', linewidth=2)
    ax3.set_title('Improvement Rate', fontsize=12, pad=15)
    ax3.set_xlabel('Generation', fontsize=10)
    ax3.set_ylabel('Fitness Change', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Plot 4: Best Solution Schedule
    best_solution = history['best_solutions'][-1]
    ax4 = plt.subplot(2, 2, 4)
    plot_schedule(best_solution.schedule, best_solution.problem.num_machines, ax4)

    plt.tight_layout(pad=3.0)
    return fig


def plot_schedule(schedule: Dict, num_machines: int, ax=None):
    """Create Gantt chart of the schedule."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # Use a colorblind-friendly palette
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6',
              '#1abc9c', '#e67e22', '#34495e', '#7f8c8d', '#16a085']
    # Extend colors if needed
    while len(colors) < num_machines:
        colors.extend(colors)
    colors = colors[:num_machines]

    # Track job positions for better text placement
    job_positions = {}

    for (job_id, op_idx), details in sorted(schedule.items(),
                                            key=lambda x: (x[1]['machine'], x[1]['start'])):
        start = details['start']
        duration = details['end'] - start
        machine = details['machine']

        # Create bar
        bar = ax.barh(y=machine, width=duration, left=start,
                      color=colors[job_id % len(colors)], alpha=0.8,
                      label=f'Job {job_id}' if job_id not in job_positions else "")

        # Store job position
        job_positions[job_id] = True

        # Add text label with improved positioning
        text_x = start + duration / 2
        text_y = machine
        ax.text(text_x, text_y, f'J{job_id}',
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='black')

    # Customize appearance
    ax.set_title('Best Schedule (Gantt Chart)', fontsize=12, pad=15)
    ax.set_xlabel('Time Units', fontsize=10)
    ax.set_ylabel('Machine', fontsize=10)
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(),
                       title='Jobs',
                       loc='center left',
                       bbox_to_anchor=(1, 0.5),
                       frameon=True,
                       fancybox=True,
                       shadow=True)
    legend.get_title().set_fontweight('bold')

    return ax