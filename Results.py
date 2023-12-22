import numpy as np
from bitstring import BitArray
import random
import math


class Results:

    def __init__(self, results_time, results, results_g_best_bin, results_g_best_float_position, results_iter_of_g_best,
                 results_parameter_fitness_index, results_accuracy_hit, results_global_min, results_global_min_value):
        self.results_time = results_time
        self.results = results
        self.results_g_best_bin = results_g_best_bin
        self.results_g_best_float_position = results_g_best_float_position
        self.results_iter_of_g_best = results_iter_of_g_best
        self.results_parameter_fitness_index = results_parameter_fitness_index
        self.results_accuracy_hit = results_accuracy_hit
        self.results_global_min = results_global_min
        self.results_global_min_value = results_global_min_value
