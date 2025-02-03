from typing import List, Dict
import numpy as np

from jsp_parser import JSPParser


class JobShopProblem:
    """Job Shop Scheduling Problem definition."""

    def __init__(self):
        self.jobs_data = []
        self.num_jobs = len(self.jobs_data)
        self.num_machines = 6

    def load_from_file(self, file_path: str, instance_name: str = None):
        with open(file_path, 'r') as f:
            content = f.read()

        instances = JSPParser.parse_file(content)

        if instance_name is None:
            # Take first instance if none specified
            instance = next(iter(instances.values()))
        else:
            if instance_name not in instances:
                raise ValueError(f"Instance '{instance_name}' not found in file")
            instance = instances[instance_name]

        # Update problem attributes
        self.jobs_data = instance.jobs_data
        self.num_jobs = instance.num_jobs
        self.num_machines = instance.num_machines
        self.instance_name = instance.name
        self.instance_description = instance.description


    def validate_solution(self, chromosome):
        """Validate if a chromosome represents a valid solution."""
        # Count operations per job
        job_counts = {}
        for job_id in chromosome:
            job_counts[job_id] = job_counts.get(job_id, 0) + 1

        # Check if each job has correct number of operations
        for job_id, job_ops in enumerate(self.jobs_data):
            if job_counts.get(job_id, 0) != len(job_ops):
                return False
        return True

    def get_operation_details(self, job_id: int, operation_idx: int) -> tuple:
        """Get machine and processing time for a specific operation."""
        return self.jobs_data[job_id][operation_idx]

    def calculate_makespan(self, schedule: dict) -> int:
        """Calculate the total makespan of a schedule."""
        return max(details['end'] for details in schedule.values())

    def __str__(self):
        """String representation of the problem instance."""
        return (f"Job Shop Problem Instance: {self.instance_name}\n"
                f"Description: {self.instance_description}\n"
                f"Jobs: {self.num_jobs}, Machines: {self.num_machines}")


class JobShopChromosome:
    """
    Represents a solution to the Job Shop Scheduling Problem.
    Each chromosome is a sequence of operations that defines the order
    in which jobs should be processed on machines.
    """
    def __init__(self, problem: JobShopProblem):
        self.problem = problem
        self.chromosome = self.create_random_chromosome()
        self.schedule = None
        self.fitness = float('inf')

    def create_random_chromosome(self) -> List[int]:
        """
        Creates a valid random sequence of jobs.
        This improved version ensures all jobs get their required number of operations
        while maintaining randomness in the sequence.
        """
        # Create a list of all required operations
        operations = []
        for job_id, job_operations in enumerate(self.problem.jobs_data):
            # Add each job_id the number of times it needs operations
            operations.extend([job_id] * len(job_operations))

        # Shuffle the operations randomly
        np.random.shuffle(operations)
        return operations

    def validate_chromosome(self) -> bool:
        """
        Validates that the chromosome represents a feasible solution.
        A chromosome is valid if each job appears exactly the number of times
        it needs operations.
        """
        # Count operations per job
        job_counts = {}
        for job_id in self.chromosome:
            job_counts[job_id] = job_counts.get(job_id, 0) + 1

        # Check against required operations for each job
        for job_id, job_operations in enumerate(self.problem.jobs_data):
            required_ops = len(job_operations)
            actual_ops = job_counts.get(job_id, 0)
            if required_ops != actual_ops:
                return False
        return True

    def decode_to_schedule(self) -> Dict:
        """
        Converts the chromosome (operation sequence) into an actual schedule.
        Returns a dictionary with timing details for each operation.
        """
        if not self.validate_chromosome():
            raise ValueError("Invalid chromosome detected during scheduling!")

        # Track when each machine and job will be available
        machine_available_time = {}  # When will each machine be free
        job_available_time = {}      # When will each job be ready for next operation
        schedule = {}                # Final schedule to return

        # Track how many operations we've scheduled for each job
        job_op_counts = {job_id: 0 for job_id in range(self.problem.num_jobs)}

        # Process each operation in the sequence
        for position, job_id in enumerate(self.chromosome):
            # Get the next operation for this job
            operation_index = job_op_counts[job_id]
            job_op_counts[job_id] += 1

            # Get machine and processing time for this operation
            machine_id, processing_time = self.problem.get_operation_details(job_id, operation_index)

            # Calculate earliest possible start time
            machine_ready = machine_available_time.get(machine_id, 0)
            job_ready = job_available_time.get(job_id, 0)
            start_time = max(machine_ready, job_ready)
            end_time = start_time + processing_time

            # Update availability times
            machine_available_time[machine_id] = end_time
            job_available_time[job_id] = end_time

            schedule[(job_id, operation_index)] = {
                'machine': machine_id,
                'start': start_time,
                'end': end_time
            }

        self.fitness = max(job_available_time.values())
        return schedule