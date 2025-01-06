import argparse
from pathlib import Path
from parameter_analyzer import ParameterAnalyzer  # Make sure to import the new analyzer


def main():
    # Set up command line argument parser with enhanced options
    parser = argparse.ArgumentParser(description='Run Job Shop Problem experiments with parameter analysis')
    parser.add_argument('input_file', help='Path to JSP instances file')
    parser.add_argument('--instance', help='Specific instance to run (optional)')
    parser.add_argument('--output-dir', default='results', help='Output directory')

    # Add parameter analysis specific arguments
    parser.add_argument('--analyze-params', action='store_true',
                        help='Run parameter analysis with multiple combinations')
    parser.add_argument('--base-population-size', type=int, default=100,
                        help='Base population size for parameter analysis')
    parser.add_argument('--base-generations', type=int, default=100,
                        help='Base number of generations for parameter analysis')

    # Keep existing arguments for single-parameter runs
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--elite-size', type=int, default=2)
    parser.add_argument('--tournament-size', type=int, default=5)
    parser.add_argument('--mutation-rate', type=float, default=0.1)
    parser.add_argument('--max-instances', type=int,
                        help='Maximum number of instances to process (default: all)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_params:
        # Use the parameter analyzer for multiple parameter combinations
        analyzer = ParameterAnalyzer(args.input_file, args.output_dir)

        if args.instance:
            import time

            # Start measuring time
            start_time = time.time()

            # Run analysis for specific instance
            print(f"\nRunning parameter analysis for instance: {args.instance}")
            results = analyzer.run_instance_analysis(args.instance)
            analysis = analyzer.analyze_results(results)
            summary_df = analyzer.save_analysis(results)

            # Stop measuring time
            elapsed_time = time.time() - start_time

            # Display summary
            print("\nParameter Comparison Summary:")
            print("=" * 80)
            print(summary_df.to_string())

            # Print elapsed time
            print(f"\nElapsed time: {elapsed_time:.2f} seconds")


        else:
            # Run analysis for multiple instances up to max_instances
            instances_to_process = analyzer.get_available_instances()
            if args.max_instances:
                instances_to_process = instances_to_process[:args.max_instances]

            for instance_name in instances_to_process:
                print(f"\nAnalyzing instance: {instance_name}")
                results = analyzer.run_instance_analysis(instance_name)
                analysis = analyzer.analyze_results(results)
                summary_df = analyzer.save_analysis(results)

                print(f"\nResults for {instance_name}:")
                print("=" * 80)
                print(summary_df.to_string())
                print("\n")

    else:
        # Use the original ExperimentRunner for single parameter runs
        from experiment_runner import ExperimentRunner

        runner = ExperimentRunner(
            input_file=args.input_file,
            output_dir=args.output_dir,
            population_size=args.population_size,
            generations=args.generations,
            elite_size=args.elite_size,
            tournament_size=args.tournament_size,
            mutation_rate=args.mutation_rate,
            max_instances=args.max_instances
        )

        if args.instance:
            results = {args.instance: runner.run_single_instance(args.instance)}
        else:
            results = runner.run_all_instances()

        runner.save_results(results)


if __name__ == "__main__":
    main()
