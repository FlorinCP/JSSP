import argparse

from configuration_runner import ConfigurationRunner


def main():
    parser = argparse.ArgumentParser(description='Run Job Shop Problem experiments')
    parser.add_argument('input_file',default="test_data.txt" ,help='Path to JSP instances file')
    parser.add_argument('--instance', required=True, help='Specific instance to run')
    parser.add_argument('--output-dir', default='results', help='Output directory')

    args = parser.parse_args()

    runner = ConfigurationRunner(
        input_file=args.input_file,
        instance_name=args.instance,
        output_dir=args.output_dir,
    )

    results = runner.run_all_combinations()

    runner.save_results(results)


if __name__ == "__main__":
    main()