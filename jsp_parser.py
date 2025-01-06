from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re


@dataclass
class JSPInstance:
    name: str
    description: str
    num_jobs: int
    num_machines: int
    jobs_data: List[List[Tuple[int, int]]]


class JSPParser:
    """Parser for Job Shop Problem instances in the standard format."""

    @staticmethod
    def parse_file(file_content: str) -> Dict[str, JSPInstance]:
        """Parse a file containing multiple JSP instances."""
        # Split content into blocks based on the instance marker pattern
        instance_blocks = re.split(r'(?=\s*instance\s+[a-zA-Z0-9]+\s*$)', file_content, flags=re.MULTILINE)

        instances = {}
        for block in instance_blocks:
            if not block.strip() or 'instance' not in block.lower():
                continue

            try:
                instance = JSPParser._parse_instance_block(block)
                if instance:
                    instances[instance.name] = instance
            except Exception as e:
                print(f"Error parsing block:\n{block[:200]}...\nError: {str(e)}")

        return instances

    @staticmethod
    def _parse_instance_block(block: str) -> Optional[JSPInstance]:
        """Parse a single instance block."""
        # Split into lines and clean
        lines = [line.strip() for line in block.split('\n') if line.strip()]

        try:
            # Get instance name
            name_match = re.search(r'instance\s+([a-zA-Z0-9]+)', lines[0])
            if not name_match:
                return None
            name = name_match.group(1)

            # Find the dimensions line (first line with exactly two numbers)
            dims_line_idx = -1
            description_lines = []

            for idx, line in enumerate(lines[1:], 1):  # Start after instance name
                # Skip separator lines
                if all(c in '+ ' for c in line):
                    continue

                # Check if this is the dimensions line
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if len(numbers) == 2:
                    dims_line_idx = idx
                    break
                elif not line.startswith('+'):  # Collect description lines
                    description_lines.append(line)

            if dims_line_idx == -1:
                raise ValueError(f"Could not find dimensions line for instance {name}")

            # Parse dimensions
            num_jobs, num_machines = map(int, lines[dims_line_idx].split())

            # Get job data lines
            data_lines = []
            current_line = dims_line_idx + 1

            while current_line < len(lines) and len(data_lines) < num_jobs:
                line = lines[current_line]
                # Skip separator lines and ensure line has numbers
                if not all(c in '+ ' for c in line) and any(c.isdigit() for c in line):
                    data_lines.append(line)
                current_line += 1

            # Parse job data
            jobs_data = []
            for line in data_lines:
                numbers = [int(x) for x in line.split()]
                # Create pairs of (machine, time)
                ops = [(numbers[i], numbers[i + 1])
                       for i in range(0, len(numbers), 2)]
                jobs_data.append(ops)

            return JSPInstance(
                name=name,
                description=' '.join(description_lines),
                num_jobs=num_jobs,
                num_machines=num_machines,
                jobs_data=jobs_data
            )

        except Exception as e:
            raise ValueError(f"Failed to parse instance {name if 'name' in locals() else 'unknown'}: {str(e)}")

    @staticmethod
    def analyze_input_file(file_path: str):
        """Analyze the input file structure and report issues."""
        print("\nAnalyzing input file structure...")

        with open(file_path, 'r') as f:
            content = f.read()

        # Check basic file properties
        print(f"File size: {len(content)} bytes")
        print(f"Number of lines: {len(content.split('\n'))}")

        # Look for instance markers
        instance_lines = [line.strip() for line in content.split('\n')
                          if line.strip().lower().startswith('instance')]
        print(f"\nFound {len(instance_lines)} instance markers:")
        for line in instance_lines[:5]:
            print(f"  - {line}")
        if len(instance_lines) > 5:
            print(f"  ... and {len(instance_lines) - 5} more")

        # Check separator lines
        separator_lines = [line.strip() for line in content.split('\n')
                           if line.strip().startswith('+')]
        print(f"\nFound {len(separator_lines)} separator lines")

        # Look for potential format issues
        print("\nChecking for potential format issues:")

        # Check for consistent separators
        if len(separator_lines) < len(instance_lines) * 2:
            print("  - Warning: Not enough separator lines for instances")

        # Check for dimension lines
        dimension_lines = [line.strip() for line in content.split('\n')
                           if re.match(r'^\d+\s+\d+$', line.strip())]
        print(f"  - Found {len(dimension_lines)} dimension lines")

        return {
            'num_instances': len(instance_lines),
            'num_separators': len(separator_lines),
            'num_dimensions': len(dimension_lines)
        }


def validate_instance(instance: JSPInstance) -> bool:
    """Validate a parsed JSP instance."""
    try:
        # Check dimensions
        if instance.num_jobs <= 0 or instance.num_machines <= 0:
            print(f"Invalid dimensions for {instance.name}: jobs={instance.num_jobs}, machines={instance.num_machines}")
            return False

        # Check job data
        if len(instance.jobs_data) != instance.num_jobs:
            print(
                f"Wrong number of jobs for {instance.name}: expected {instance.num_jobs}, got {len(instance.jobs_data)}")
            return False

        # Check operations per job
        for job_idx, job in enumerate(instance.jobs_data):
            if len(job) != instance.num_machines:
                print(f"Wrong number of operations in job {job_idx} for {instance.name}")
                return False

            # Check machine numbers and processing times
            for op_idx, (machine, time) in enumerate(job):
                if machine < 0 or machine >= instance.num_machines:
                    print(f"Invalid machine number in job {job_idx}, operation {op_idx} for {instance.name}")
                    return False
                if time <= 0:
                    print(f"Invalid processing time in job {job_idx}, operation {op_idx} for {instance.name}")
                    return False

        return True

    except Exception as e:
        print(f"Error validating {instance.name}: {str(e)}")
        return False