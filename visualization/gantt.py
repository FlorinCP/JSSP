from typing import Dict
from .utils import get_color_palette


def create_gantt_chart(ax, schedule: Dict, num_machines: int):
    """Create Gantt chart visualization of the schedule."""
    colors = get_color_palette(num_machines)
    job_positions = {}

    for (job_id, op_idx), details in sorted(
            schedule.items(),
            key=lambda x: (x[1]['machine'], x[1]['start'])):
        start = details['start']
        duration = details['end'] - start
        machine = details['machine']

        # Create bar
        ax.barh(y=machine, width=duration, left=start,
                color=colors[job_id % len(colors)], alpha=0.8,
                label=f'Job {job_id}' if job_id not in job_positions else "")

        # Store job position and add label
        job_positions[job_id] = True
        ax.text(start + duration / 2, machine, f'J{job_id}',
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='black')

    # Customize appearance
    ax.set_title('Best Schedule (Gantt Chart)', pad=15)
    ax.set_xlabel('Time Units')
    ax.set_ylabel('Machine')
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add legend
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