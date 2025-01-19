import json
from pathlib import Path
from typing import Dict, Any


def extract_important_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract important information from simulation results.

    This function processes the raw simulation data and keeps only the most
    relevant metrics for analysis.
    """
    important_info = {}

    for instance_name, instance_data in data.items():
        instance_info = {
            "instance_info": {
                "num_jobs": instance_data["problem_info"]["num_jobs"],
                "num_machines": instance_data["problem_info"]["num_machines"],
                "description": instance_data["problem_info"]["description"]
            },
            "parameter_results": {}
        }

        # Process results for each parameter set
        for param_set, param_data in instance_data["parameter_sets"].items():
            param_results = {
                "configuration": {
                    k: v for k, v in param_data["parameters"].items()
                    if k not in ["name"]  # Exclude redundant name field
                },
                "performance": {
                    "best_fitness": param_data["stats"]["best_fitness"],
                    "improvement_percentage": param_data["stats"]["improvement_percentage"],
                    "convergence_generation": param_data["stats"]["convergence_generation"],
                    "total_generations": param_data["stats"]["total_generations"],
                    "final_diversity": param_data["stats"]["final_diversity"]
                }
            }

            instance_info["parameter_results"][param_set] = param_results

        important_info[instance_name] = instance_info

    return important_info


def process_results_file(input_path: str, output_path: str):
    """
    Process a results file and save simplified version.

    Args:
        input_path: Path to the input JSON file
        output_path: Path where to save the processed results
    """
    try:
        # Read input file
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Extract important information
        processed_data = extract_important_info(data)

        # Save processed results
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)

        print(f"Successfully processed results and saved to {output_path}")

    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main():
    """Main function to run the script."""
    # Create results directory if it doesn't exist
    output_dir = Path("processed_results")
    output_dir.mkdir(exist_ok=True)

    # Process all JSON files in current directory
    for input_file in Path("overall_results").glob("*.json"):
        output_file = output_dir / f"processed_{input_file.name}"
        print(f"\nProcessing {input_file}...")
        process_results_file(str(input_file), str(output_file))


if __name__ == "__main__":
    main()