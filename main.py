import argparse

from simulation_runner import SimulationRunner


def main():
    parser = argparse.ArgumentParser(description='Run Job Shop Problem experiments')
    parser.add_argument('input_file', help='Path to JSP instances file')
    parser.add_argument('--instance', help='Specific instance to run (optional)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--visualize', action='store_true', help='Render charts')
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--elite-size', type=int, default=2)
    parser.add_argument('--tournament-size', type=int, default=5)
    parser.add_argument('--mutation-rate', type=float, default=0.1)
    parser.add_argument('--multiple-params', action='store_true',
                        help='Run multiple parameter configurations')
    parser.add_argument('--max-instances', type=int,
                        help='Maximum number of instances to process (default: all)')

    args = parser.parse_args()

    runner = SimulationRunner(
        input_file=args.input_file,
        output_dir=args.output_dir,
        population_size=args.population_size,
        generations=args.generations,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        mutation_rate=args.mutation_rate,
        max_instances=args.max_instances
    )

    runner.run(args.instance, args.multiple_params)


if __name__ == "__main__":
    main()