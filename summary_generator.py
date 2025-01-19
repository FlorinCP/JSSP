import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats


class InstanceAnalyzer:
    """Analyzes job shop instances and their characteristics."""

    def __init__(self):
        self.instance_categories = {
            'tiny': {'jobs': (0, 10), 'machines': (0, 5)},
            'small': {'jobs': (0, 15), 'machines': (5, 10)},
            'medium': {'jobs': (15, 30), 'machines': (10, 15)},
            'large': {'jobs': (30, 50), 'machines': (15, 20)},
            'extra_large': {'jobs': (50, float('inf')), 'machines': (20, float('inf'))}
        }

    def categorize_instance(self, instance_name: str, num_jobs: int, num_machines: int) -> str:
        """Categorize instance based on size and characteristics."""
        # First check standard instance families
        if instance_name.startswith('ft'):
            return 'fisher_thompson'
        elif instance_name.startswith('la'):
            return 'lawrence'
        elif instance_name.startswith('abz'):
            return 'adams_balas_zawack'
        elif instance_name.startswith('swv'):
            return 'storer_wu_vaccari'
        elif instance_name.startswith('yn'):
            return 'yamada_nakano'
        elif instance_name.startswith('orb'):
            return 'orbiter'

        # If not a standard family, categorize by size
        for category, limits in self.instance_categories.items():
            if (limits['jobs'][0] <= num_jobs < limits['jobs'][1] and
                    limits['machines'][0] <= num_machines < limits['machines'][1]):
                return category

        return 'uncategorized'


class ParameterAnalyzer:
    """Analyzes the effectiveness of different parameter sets across instance categories."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.instance_analyzer = InstanceAnalyzer()
        self.data = pd.DataFrame()
        self.parameter_impacts = {}

    def load_instance_data(self, instance_path: str) -> Tuple[int, int]:
        """Extract number of jobs and machines from instance file."""
        try:
            with open(instance_path, 'r') as f:
                content = f.read()
                # Look for dimensions line (e.g., "20 15" for 20 jobs, 15 machines)
                lines = content.split('\n')
                for line in lines:
                    if len(line.split()) == 2:
                        try:
                            jobs, machines = map(int, line.split())
                            return jobs, machines
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Error reading instance file {instance_path}: {str(e)}")
        return None, None

    def load_all_data(self):
        """Load and process all parameter comparison data."""
        all_data = []

        for instance_dir in self.results_dir.glob("*"):
            if instance_dir.is_dir():
                csv_path = instance_dir / "parameter_comparison.csv"
                if csv_path.exists():
                    try:
                        # Load parameter comparison data
                        df = pd.read_csv(csv_path)
                        df['Instance'] = instance_dir.name

                        # Load instance characteristics
                        instance_file = Path("test_data.txt")  # Assuming this is where instances are
                        if instance_file.exists():
                            num_jobs, num_machines = self.load_instance_data(str(instance_file))
                            if num_jobs and num_machines:
                                df['Num_Jobs'] = num_jobs
                                df['Num_Machines'] = num_machines
                                df['Instance_Category'] = df.apply(
                                    lambda x: self.instance_analyzer.categorize_instance(
                                        x['Instance'], x['Num_Jobs'], x['Num_Machines']
                                    ),
                                    axis=1
                                )
                                all_data.append(df)
                    except Exception as e:
                        print(f"Error processing {csv_path}: {str(e)}")
                        continue

        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)

    def analyze_parameter_impacts(self):
        """Analyze how different parameters impact performance across instance categories."""
        impacts = {}

        # Analyze each parameter's correlation with improvement
        parameters = ['Population Size', 'Generations', 'Mutation Rate']
        categories = self.data['Instance_Category'].unique()

        for category in categories:
            category_data = self.data[self.data['Instance_Category'] == category]

            category_impacts = {}
            for param in parameters:
                correlation = stats.pearsonr(
                    category_data[param],
                    category_data['Improvement %']
                )
                category_impacts[param] = {
                    'correlation': correlation[0],
                    'p_value': correlation[1]
                }

            impacts[category] = category_impacts

        self.parameter_impacts = impacts
        return impacts

    def identify_best_parameters(self) -> Dict:
        """Identify best parameter sets for each instance category."""
        best_params = {}

        for category in self.data['Instance_Category'].unique():
            category_data = self.data[self.data['Instance_Category'] == category]

            # Get best parameter set based on average improvement
            best_set = category_data.groupby('Parameter Set')['Improvement %'].mean().idxmax()

            # Get average metrics for best set
            best_metrics = category_data[category_data['Parameter Set'] == best_set].mean()

            best_params[category] = {
                'parameter_set': best_set,
                'avg_improvement': best_metrics['Improvement %'],
                'avg_convergence': best_metrics['Convergence Gen'],
                'final_diversity': best_metrics['Final Diversity']
            }

        return best_params

    def generate_visualizations(self):
        """Generate comprehensive visualizations of parameter impacts."""
        figures = []

        # 1. Parameter impact by instance category
        plt.figure(figsize=(15, 8))
        impact_data = []
        for category, impacts in self.parameter_impacts.items():
            for param, metrics in impacts.items():
                impact_data.append({
                    'Category': category,
                    'Parameter': param,
                    'Correlation': metrics['correlation']
                })

        impact_df = pd.DataFrame(impact_data)
        sns.heatmap(
            impact_df.pivot(index='Category', columns='Parameter', values='Correlation'),
            annot=True, cmap='RdYlBu', center=0
        )
        plt.title('Parameter Impact Correlation by Instance Category')
        plt.tight_layout()
        figures.append(plt.gcf())

        # 2. Performance distribution across instance categories
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=self.data,
            x='Instance_Category',
            y='Improvement %',
            hue='Parameter Set'
        )
        plt.xticks(rotation=45)
        plt.title('Performance Distribution by Instance Category')
        plt.tight_layout()
        figures.append(plt.gcf())

        # 3. Convergence patterns
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=self.data,
            x='Num_Jobs',
            y='Convergence Gen',
            hue='Parameter Set',
            style='Instance_Category',
            size='Improvement %'
        )
        plt.title('Convergence Patterns by Problem Size')
        plt.tight_layout()
        figures.append(plt.gcf())

        return figures

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        best_params = self.identify_best_parameters()

        report = []
        report.append("Job Shop Scheduling Parameter Analysis Report")
        report.append("=" * 50 + "\n")

        # Overall statistics
        report.append("1. Dataset Overview")
        report.append("-" * 30)
        report.append(f"Total instances analyzed: {len(self.data['Instance'].unique())}")
        report.append(f"Instance categories: {', '.join(sorted(self.data['Instance_Category'].unique()))}")
        report.append("")

        # Best parameters by category
        report.append("2. Best Parameters by Instance Category")
        report.append("-" * 30)
        for category, params in best_params.items():
            report.append(f"\n{category}:")
            report.append(f"  Best parameter set: {params['parameter_set']}")
            report.append(f"  Average improvement: {params['avg_improvement']:.2f}%")
            report.append(f"  Average convergence generation: {params['avg_convergence']:.1f}")

        # Parameter impact analysis
        report.append("\n3. Parameter Impact Analysis")
        report.append("-" * 30)
        for category, impacts in self.parameter_impacts.items():
            report.append(f"\n{category}:")
            for param, metrics in impacts.items():
                significance = "significant" if metrics['p_value'] < 0.05 else "not significant"
                report.append(
                    f"  {param}: correlation = {metrics['correlation']:.3f} ({significance})"
                )

        # Key findings and recommendations
        report.append("\n4. Key Findings and Recommendations")
        report.append("-" * 30)

        # Add specific findings based on the analysis
        correlations = pd.DataFrame(self.parameter_impacts).transpose()
        strong_correlations = []
        for param in correlations.columns:
            categories = correlations[correlations[param] > 0.5].index.tolist()
            if categories:
                strong_correlations.append(
                    f"{param} has strong positive impact on: {', '.join(categories)}"
                )

        report.extend(strong_correlations)

        return "\n".join(report)


def main():
    analyzer = ParameterAnalyzer("overall_results")

    print("Loading and processing data...")
    analyzer.load_all_data()

    print("Analyzing parameter impacts...")
    analyzer.analyze_parameter_impacts()

    print("Generating visualizations...")
    figures = analyzer.generate_visualizations()
    for i, fig in enumerate(figures):
        fig.savefig(f'parameter_analysis_{i + 1}.png', dpi=300, bbox_inches='tight')

    print("Generating report...")
    report = analyzer.generate_report()
    with open('parameter_analysis_report.txt', 'w') as f:
        f.write(report)

    print("Analysis complete. Check the generated files for results.")


if __name__ == "__main__":
    main()