import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os


class GAVisualizer:
    """
    A class for creating comprehensive visualizations of Genetic Algorithm results.
    """

    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not os.access(self.output_dir, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {self.output_dir}")

        # Set style for all plots
        plt.style.use('seaborn-v0_8-bright')
        sns.set_palette("husl")

    def create_dashboard(self, results_data):
        """
        Create a complete dashboard of visualizations for the GA results.
        """
        # Create a figure with a grid layout
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f'Genetic Algorithm Analysis Dashboard - Instance: {results_data["instance_name"]}',
                     fontsize=16, y=0.95)

        # Define grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1.5])

        # Create subplots
        ax_kpi = fig.add_subplot(gs[0, :])
        ax_fitness = fig.add_subplot(gs[1, 0])
        ax_diversity = fig.add_subplot(gs[1, 1])
        ax_evolution = fig.add_subplot(gs[2, 0])
        ax_comparison = fig.add_subplot(gs[2, 1])

        # Generate individual plots
        self._plot_kpi_summary(results_data, ax_kpi)
        self._plot_fitness_comparison(results_data, ax_fitness)
        self._plot_diversity_comparison(results_data, ax_diversity)
        self._plot_evolution(results_data, ax_evolution)
        self._plot_parameter_comparison(results_data, ax_comparison)

        # Adjust layout
        plt.tight_layout()

        # Save dashboard
        instance_name = results_data["instance_name"]
        plt.savefig(self.output_dir / f"ga_dashboard_{instance_name}.png", dpi=300, bbox_inches='tight')
        os.sync()
        plt.close()

    def _plot_kpi_summary(self, results_data, ax):
        """Create a summary of key performance indicators."""
        ax.axis('off')

        # Find best configuration
        best_config = min(results_data["parameter_combinations"].items(),
                          key=lambda x: x[1]["stats"]["best_fitness"])

        # Prepare KPI text
        kpi_text = (
            f'Best Fitness: {best_config[1]["stats"]["best_fitness"]:.2f}'
            f' (Configuration: {best_config[0]})\n'
            f'Maximum Improvement: {best_config[1]["stats"]["improvement_percentage"]:.2f}%\n'
            f'Population Size: {best_config[1]["parameters"]["population_size"]}\n'
            f'Generations: {best_config[1]["stats"]["total_generations"]}'
        )

        # Add text box with KPIs
        ax.text(0.5, 0.5, kpi_text,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                ha='center', va='center', fontsize=12)

    def _plot_fitness_comparison(self, results_data, ax):
        """Create bar plot comparing best fitness across configurations with custom colors."""

        # Extract configurations and fitness values
        configs = list(results_data["parameter_combinations"].keys())
        fitness_values = [
            results_data["parameter_combinations"][config]["stats"]["best_fitness"]
            for config in configs
        ]

        # Identify the best (lowest) and worst (highest) fitness values
        best_index = fitness_values.index(min(fitness_values))
        worst_index = fitness_values.index(max(fitness_values))

        # Create a visually appealing color palette
        colors = ['lightgray'] * len(fitness_values)  # Default color for all bars
        colors[best_index] = '#2ca02c'  # Vibrant green for the best bar
        colors[worst_index] = '#d62728'  # Bright red for the worst bar

        # Plot the bar chart with the custom color palette
        sns.barplot(x=configs, y=fitness_values, ax=ax, palette=colors)
        ax.set_title('Best Fitness Comparison')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel('Fitness Value')

        # Optionally annotate the best and worst bars
        ax.text(
            best_index, fitness_values[best_index] + 0.02,
            f'{fitness_values[best_index]:.2f}',
            ha='center', va='bottom', fontsize=10, color='#2ca02c', fontweight='bold'
        )
        ax.text(
            worst_index, fitness_values[worst_index] + 0.02,
            f'{fitness_values[worst_index]:.2f}',
            ha='center', va='bottom', fontsize=10, color='#d62728', fontweight='bold'
        )

    def _plot_diversity_comparison(self, results_data, ax):
        """Create scatter plot of improvement vs diversity."""
        improvements = []
        diversities = []
        names = []

        for config_name, config_data in results_data["parameter_combinations"].items():
            improvements.append(config_data["stats"]["improvement_percentage"])
            diversities.append(config_data["stats"]["final_diversity"])
            names.append(config_name)

        scatter = ax.scatter(diversities, improvements)
        ax.set_title('Improvement vs Final Diversity')
        ax.set_xlabel('Final Diversity')
        ax.set_ylabel('Improvement Percentage')

        # Add labels to points
        for i, name in enumerate(names):
            ax.annotate(name, (diversities[i], improvements[i]))

    def _plot_evolution(self, results_data, ax):
        """Plot evolution of best configuration."""
        best_config = min(results_data["parameter_combinations"].items(),
                          key=lambda x: x[1]["stats"]["best_fitness"])

        generations = range(len(best_config[1]["history"]["best_fitness"]))
        fitness_values = best_config[1]["history"]["best_fitness"]
        diversity_values = best_config[1]["history"]["diversity"]

        ax2 = ax.twinx()

        line1 = ax.plot(generations, fitness_values, 'b-', label='Best Fitness')
        line2 = ax2.plot(generations, diversity_values, 'r-', label='Diversity')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness', color='b')
        ax2.set_ylabel('Diversity', color='r')

        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')

        ax.set_title(f'Evolution of Best Configuration ({best_config[0]})')

    def _plot_parameter_comparison(self, results_data, ax):
        """Create parallel coordinates plot for parameter comparison."""
        # Prepare data for parallel coordinates
        param_data = []
        for config_name, config_data in results_data["parameter_combinations"].items():
            params = config_data["parameters"]
            stats = config_data["stats"]
            param_data.append({
                'name': config_name,
                'population_size': params["population_size"],
                'tournament_size': params["tournament_size"],
                'mutation_rate': params["mutation_rate"],
                'best_fitness': stats["best_fitness"]
            })

        # Create parallel coordinates plot
        for data in param_data:
            values = [data['population_size'], data['tournament_size'],
                      data['mutation_rate'], data['best_fitness']]
            ax.plot([0, 1, 2, 3], values, '-o', label=data['name'])

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['Population\nSize', 'Tournament\nSize',
                            'Mutation\nRate', 'Best\nFitness'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title('Parameter Comparison')


def create_visualization(results_file, output_dir='results'):
    """
    Create visualizations from a results JSON file.

    Args:
        results_file (str): Path to the JSON file containing GA results
        output_dir (str): Directory to save the visualizations
    """
    with open(results_file, 'r') as f:
        results_data = json.load(f)

    visualizer = GAVisualizer(output_dir)
    visualizer.create_dashboard(results_data)

