from typing import List
from main import JobShopProblem


class JobShopChromosome:
    """
    This class represents one possible solution to our scheduling problem.
    Think of it like a DNA sequence that tells us the order to process our jobs.
    """

    def __init__(self, problem: JobShopProblem):
        self.problem = problem
        self.chromosome = self.create_random_chromosome()
        self.fitness = float('inf')

    def create_random_chromosome(self) -> List[int]:
        """
        Creates a random valid sequence of jobs.
        Example: If Job 0 needs 3 operations and Job 1 needs 2 operations,
        we might create: [0,1,0,1,0]
        """
        # Start with an empty list
        operations = []

        # For each job in our problem...
        for job_id, job_operations in enumerate(self.problem.jobs_data):

            # How many operations does this job need?
            num_operations = len(job_operations)

            # Add this job's ID that many times to our list
            operations.extend([job_id] * num_operations)
            print(f"Added {num_operations} operations for Job {job_id}")
            print(f"Current operations list: {operations}")

        # Shuffle the list randomly to create our chromosome
        np.random.shuffle(operations)
        print(f"Final chromosome after shuffling: {operations}")
        return operations

    def validate_chromosome(self) -> bool:
        """
        Checks if our chromosome is valid (has correct number of operations for each job).
        Returns True if valid, False if not.
        """
        # Count how many times each job appears
        job_counts = {}
        for job_id in self.chromosome:
            job_counts[job_id] = job_counts.get(job_id, 0) + 1

        # Check against how many operations each job should have
        for job_id, job_operations in enumerate(self.problem.jobs_data):
            required_ops = len(job_operations)                              # How many operations job should have
            actual_ops = job_counts.get(job_id, 0)                          # How many it has in chromosome
            if required_ops != actual_ops:
                print(f"Invalid chromosome! Job {job_id} needs {required_ops} "
                      f"operations but has {actual_ops}")
                return False
        return True

    def decode_to_schedule(self) -> Dict:
        """
        Converts our chromosome (sequence of jobs) into an actual schedule.
        Returns a dictionary telling us when each operation starts and ends.
        """

        # First, make sure our chromosome is valid
        if not self.validate_chromosome():
            raise ValueError("Invalid chromosome!")

        # These dictionaries help us track timing:
        machine_available_time = {}                 # When will each machine be free?
        job_available_time = {}                     # When will each job be ready for its next operation?
        schedule = {}                               # Our final schedule (what we'll return)

        print("\nDecoding chromosome into schedule:")
        print(f"Chromosome sequence: {self.chromosome}")

        # Process each job in our sequence
        for position, job_id in enumerate(self.chromosome):

            # Figure out which operation this is for this job
            # (how many times have we seen this job so far?)
            job_occurrences = self.chromosome[:position + 1].count(job_id)
            operation_index = job_occurrences - 1

            # Get the details for this operation
            machine_id, processing_time = self.problem.jobs_data[job_id][operation_index]

            # When can we start this operation?
            machine_ready = machine_available_time.get(machine_id, 0)    # When machine is free
            job_ready = job_available_time.get(job_id, 0)                # When job is ready
            start_time = max(machine_ready, job_ready)                   # Start at later of these times
            end_time = start_time + processing_time                      # Calculate when we'll finish

            # Print detailed information about what we're scheduling
            print(f"\nScheduling Job {job_id}, Operation {operation_index}:")
            print(f"  Using Machine {machine_id} for {processing_time} time units")
            print(f"  Machine ready at: {machine_ready}")
            print(f"  Job ready at: {job_ready}")
            print(f"  Will start at: {start_time}")
            print(f"  Will finish at: {end_time}")

            # Update our tracking information
            machine_available_time[machine_id] = end_time  # Machine will be busy until end_time
            job_available_time[job_id] = end_time  # Job can't continue until end_time

            # Store this operation in our schedule
            schedule[(job_id, operation_index)] = {
                'machine': machine_id,
                'start': start_time,
                'end': end_time
            }

        # The fitness is the total time needed (when does the last operation finish?)
        self.fitness = max(job_available_time.values())
        print(f"\nFinal schedule completion time: {self.fitness}")
        return schedule
