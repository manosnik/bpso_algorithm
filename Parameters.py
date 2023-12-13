import numpy as np
from bitstring import BitArray
import random
import math

class Parameters:
    def __init__(self,size, ff_code, number_of_decimals, omega_generator, c1_generator, c2_generator,
                 mutation_mode, v_ones_max_percentage, algorithm_iterations, dim_of_multidim_fun):
        self.size = size
        self.ff_code = ff_code
        self.number_of_decimals = number_of_decimals

        self.omega_generator = omega_generator
        self.c1_generator = c1_generator
        self.c2_generator = c2_generator

        self.mutation_mode = mutation_mode
        self.v_ones_max_percentage = v_ones_max_percentage

        self.algorithm_iterations = algorithm_iterations

        self.dim_of_multidim_fun = dim_of_multidim_fun
