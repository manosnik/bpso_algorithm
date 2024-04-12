import numpy as np
from parameters import Parameters


class Results:

    # Initialize the Results object with empty lists to store results.
    # Args:
    # - parameters (Parameters): An instance of the Parameters class.

    def __init__(self, parameters: Parameters):
        self.gbest_value = []  # List to store the global best values.
        self.duration = []  # List to store the duration of each optimization run.
        self.gbest_bin = []  # List to store the binary representation of the global best solution.
        self.gbest_float_position = []  # List to store the float position of the global best solution.
        self.iter_of_gbest = []  # List to store the iteration number when the global best solution was found.

        self.parameter_fitness_index = []  # List to store fitness indexes metric .
        self.accuracy_hit_array = []  # List to store accuracy hits.
        self.global_min = None  # Global minimum position
        self.global_min_value = None  # Global minimum value.

        self.directivity = []  # List to store directivity values when we are on electromagnetic problem.

        self.sll = []  # List to store side-lobe level values when we are on electromagnetic problem.

        array_size = len(parameters.param_dict())  # Compute size of optimum parameter set arrrays

        self.data_array = []  # Array to store the valuable data results after optimization

        self.optimum_parameter_set_index_based = np.empty(array_size)  # Array to store the optimum parameter set based on index metric.

        self.optimum_parameter_set_accuracy_based = np.empty(array_size)  # Array to store the optimum parameter set based on accuracy.

        # If parameter selection mode is off we don;t need these arrays(but must be declared)
        if parameters.parameters_selection_mode == 1:
            self.optimum_parameter_set_index_based[-2] = np.inf
            self.optimum_parameter_set_accuracy_based[-1] = -1

    def clear_arrays(self):
        # Clear all result arrays needed.
        self.gbest_value.clear()
        self.duration.clear()
        self.gbest_bin.clear()
        self.gbest_float_position.clear()
        self.iter_of_gbest.clear()
        self.parameter_fitness_index.clear()
        self.accuracy_hit_array.clear()

        self.global_min = None
        self.global_min_value = None
        return


class ResultsRay:

    # Class to store results obtained during the optimization process when using Ray for parallel processing.

    def __init__(self, duration, gbest_value, gbest_bin, gbest_float_position, iter_of_gbest,
                 global_min, global_min_value, parameter_fitness_index=None, accuracy_hit=None,
                 directivity=None, sll=None):
        self.duration = duration
        self.gbest_value = gbest_value
        self.gbest_bin = gbest_bin
        self.gbest_float_position = gbest_float_position
        self.iter_of_gbest = iter_of_gbest
        self.global_min = global_min
        self.global_min_value = global_min_value
        self.parameter_fitness_index = parameter_fitness_index
        self.accuracy_hit = accuracy_hit
        self.directivity = directivity
        self.sll = sll
