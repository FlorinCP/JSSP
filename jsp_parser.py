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
    """Parser for Job Shop Problem instances."""

    @staticmethod
    def _parse_instance_block(block: str) -> Optional[JSPInstance]:
        """
        Parse a single instance block with improved dimension handling.
        """
        # Split into lines and remove empty ones
        lines = [line.strip() for line in block.split('\n') if line.strip()]

        try:
            # Get instance name
            name_match = re.search(r'instance\s+([a-zA-Z0-9]+)', lines[0])
            if not name_match:
                return None
            name = name_match.group(1)

            # Find the dimensions line
            dims_line_idx = -1
            description_lines = []

            for idx, line in enumerate(lines[1:], 1):
                # Skip separator lines
                if all(c in '+ ' for c in line):
                    continue

                # Look for a line with exactly two numbers and nothing else
                stripped_line = line.strip()
                numbers = [num for num in stripped_line.split() if num.isdigit()]

                if len(numbers) == 2 and all(c.isdigit() or c.isspace() for c in stripped_line):
                    dims_line_idx = idx
                    break
                elif not line.startswith('+'):
                    description_lines.append(line)

            if dims_line_idx == -1:
                raise ValueError(f"Could not find dimensions line for instance {name}")

            # Parse dimensions
            num_jobs, num_machines = map(int, lines[dims_line_idx].split())

            # Parse job data
            jobs_data = []
            current_line = dims_line_idx + 1

            while current_line < len(lines) and len(jobs_data) < num_jobs:
                line = lines[current_line].strip()

                # Skip separator lines and ensure line has numbers
                if not all(c in '+ ' for c in line) and any(c.isdigit() for c in line):
                    # Extract all numbers from the line
                    numbers = [int(n) for n in line.split()]

                    # Create pairs of (machine, time)
                    operations = [(numbers[i], numbers[i + 1])
                                  for i in range(0, len(numbers), 2)]

                    if len(operations) != num_machines:
                        raise ValueError(
                            f"Wrong number of operations in job {len(jobs_data)} "
                            f"for instance {name}"
                        )

                    jobs_data.append(operations)
                current_line += 1

            # Validate job count
            if len(jobs_data) != num_jobs:
                raise ValueError(
                    f"Wrong number of jobs for {name}: "
                    f"expected {num_jobs}, got {len(jobs_data)}"
                )

            return JSPInstance(
                name=name,
                description=' '.join(description_lines),
                num_jobs=num_jobs,
                num_machines=num_machines,
                jobs_data=jobs_data
            )

        except Exception as e:
            raise ValueError(f"Failed to parse instance {name}: {str(e)}")

    @staticmethod
    def parse_file(file_content: str) -> Dict[str, JSPInstance]:
        """Parse all instances from a file."""
        # Split content into blocks based on instance markers
        instance_blocks = re.split(
            r'(?=\s*instance\s+[a-zA-Z0-9]+\s*$)',
            file_content,
            flags=re.MULTILINE
        )

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