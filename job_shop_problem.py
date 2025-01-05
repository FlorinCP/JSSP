class JobShopProblem:
    """
    This class represents a Job Shop Scheduling Problem.
    Think of it like managing a workshop where different jobs need to use different machines
    in a specific order, like a car going through different stations in a repair shop.
    """

    def __init__(self):
        # jobs_data stores our manufacturing instructions
        # For each job, we have a list of tuples: (machine_number, time_needed)
        # Example: [(0,3), (1,2)] means:
        #   - First use machine 0 for 3 time units
        #   - Then use machine 1 for 2 time units
        self.jobs_data = [
            [(0, 3), (1, 2), (2, 2)],  # Job 0's sequence of operations
            [(0, 2), (2, 1), (1, 4)],  # Job 1's sequence of operations
            [(1, 4), (2, 3)]           # Job 2's sequence of operations
        ]

        # Count how many jobs and machines we have
        self.num_jobs = len(self.jobs_data)  # How many different jobs
        self.num_machines = 3                # How many machines available