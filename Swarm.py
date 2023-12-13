from Particle import Particle
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor


class Swarm:
    def __init__(self, size, ff_code, number_of_decimals, omega_generator, c1_generator, c2_generator, mutation_mode,
                 v_ones_max_percentage, dim_of_multidim_fun):

        # self.bound_min = None
        self.size = size
        self.ff_code = ff_code
        self.number_of_decimals = number_of_decimals

        self.omega_generator = omega_generator
        self.c1_generator = c1_generator
        self.c2_generator = c2_generator

        self.mutation_mode = mutation_mode
        self.v_ones_max_percentage = v_ones_max_percentage

        self.population = []
        self.gbest_value = np.inf

        self.dim_of_multidim_fun = dim_of_multidim_fun

        self.choose_ff()

        for i in range(self.size):
            p = Particle(i, self.ff_code, self.bound_min, self.bound_max, self.number_of_decimals, self.omega_generator,
                         self.c1_generator, self.c2_generator, self.mutation_mode, self.v_ones_max_percentage)
            self.population.append(p)

    def __str__(self):
        for i in range(self.size):
            print(self.population[i])
        print(f"gbest: {self.g_best.bin}\tgbest_fitness_value :{self.gbest_value}")
        return "-------------------------------------------------------------------------------------------------------------"


    #TODO add glodal_min attribute
    def choose_ff(self):
        if self.ff_code == 1:
            self.ff_name = "Eggholder"
            self.bound_min = [-512, -512]
            self.bound_max = [512, 512]
        elif self.ff_code == 2:
            self.ff_name = "Rastring"
            self.bound_min = [-5.12, -5.12]
            self.bound_max = [5.12, 5.12]
        elif self.ff_code == 3:
            self.ff_name = "Ackley"
            self.bound_min = [-32.768] * self.dim_of_multidim_fun
            self.bound_max = [32.768] * self.dim_of_multidim_fun
        elif self.ff_code == 4:
            self.ff_name = "Rosenbrock"
            self.bound_min = [-5] * self.dim_of_multidim_fun
            self.bound_max = [10] * self.dim_of_multidim_fun
        elif self.ff_code == 5:
            self.ff_name = "Dropwave"
            self.bound_min = [-5.12, -5.12]
            self.bound_max = [5.12, 5.12]
        elif self.ff_code == 6:
            self.ff_name = "Bukin N.6"
            self.bound_min = [-15, -3]
            self.bound_max = [-5, 3]
            self.global_min=[-10,1]
        elif self.ff_code == 7:
            self.ff_name = "De Jong N.5"
            self.bound_min = [-65.536, -65.536]
            self.bound_max = [65.536, 65.536]
        elif self.ff_code == 8:
            self.ff_name = "Schwefel"
            self.bound_min = [-500] * self.dim_of_multidim_fun
            self.bound_max = [500] * self.dim_of_multidim_fun
        elif self.ff_code == 9:
            self.ff_name = "Cross-in-Tray"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
        elif self.ff_code == 10:
            self.ff_name = "Holder Table"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
        elif self.ff_code == 11:
            self.ff_name = "Bohachevsky"
            self.bound_min = [-100, -100]
            self.bound_max = [100, 100]
        elif self.ff_code == 12:
            self.ff_name = "Griewank"
            self.bound_min = [-600] * self.dim_of_multidim_fun
            self.bound_max = [600] * self.dim_of_multidim_fun
        elif self.ff_code == 13:
            self.ff_name = "Easom"
            self.bound_min = [-100, -100]
            self.bound_max = [100, 100]
        elif self.ff_code == 14:
            self.ff_name = "Dixon-Price"
            self.bound_min = [-10] * self.dim_of_multidim_fun
            self.bound_max = [10] * self.dim_of_multidim_fun
        elif self.ff_code == 15:
            self.ff_name = "Six-Hump Camel"
            self.bound_min = [-3, -2]
            self.bound_max = [3, 2]
        elif self.ff_code == 16:
            self.ff_name = "Gramacy & Lee Function"
            self.bound_min = [0.5]
            self.bound_max = [2.5]
        elif self.ff_code == 17:
            self.ff_name = "Shubert"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
        elif self.ff_code == 18:
            self.ff_name = "Langermann"
            self.bound_min = [0, 0]
            self.bound_max = [10, 10]
        elif self.ff_code == 19:
            self.ff_name = "Schaffer N.2"
            self.bound_min = [-100, -100]
            self.bound_max = [100, 100]
        elif self.ff_code == 20:
            self.ff_name = "Trid"
            self.bound_min = [-(self.dim_of_multidim_fun ** 2)] * self.dim_of_multidim_fun
            self.bound_max = [self.dim_of_multidim_fun ** 2] * self.dim_of_multidim_fun
        elif self.ff_code == 21:
            self.ff_name = "Sphere"
            self.bound_min = [-5.12] * self.dim_of_multidim_fun
            self.bound_max = [5.12] * self.dim_of_multidim_fun
        elif self.ff_code == 22:
            self.ff_name = "Sum Squares function"
            self.bound_min = [-10] * self.dim_of_multidim_fun
            self.bound_max = [10] * self.dim_of_multidim_fun
        elif self.ff_code == 23:
            self.ff_name = "Sum of Different Powers function"
            self.bound_min = [-1] * self.dim_of_multidim_fun
            self.bound_max = [1] * self.dim_of_multidim_fun
        elif self.ff_code == 24:
            self.ff_name = "Rotated Hyper-Ellipsoid"
            self.bound_min = [-65.536] * self.dim_of_multidim_fun
            self.bound_max = [65.536] * self.dim_of_multidim_fun
        elif self.ff_code == 25:
            self.ff_name = "Booth"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
        elif self.ff_code == 26:
            self.ff_name = "Matyas"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
        elif self.ff_code == 27:
            self.ff_name = "McCormick"
            self.bound_min = [-1.5, -3]
            self.bound_max = [4, 4]
        elif self.ff_code == 28:
            self.ff_name = "Levy"
            self.bound_min = [-10] * self.dim_of_multidim_fun
            self.bound_max = [10] * self.dim_of_multidim_fun
        elif self.ff_code == 29:
            self.ff_name = "Levy N.13"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
        elif self.ff_code == 30:
            self.ff_name = "Schaffer N.4"
            self.bound_min = [-100, -100]
            self.bound_max = [100, 100]
        elif self.ff_code == 31:
            self.ff_name = "Power Sum function"
            self.bound_min = [0, 0, 0, 0]
            self.bound_max = [4, 4, 4, 4]
        elif self.ff_code == 32:
            self.ff_name = "Zakharov"
            self.bound_min = [-5] * self.dim_of_multidim_fun
            self.bound_max = [10] * self.dim_of_multidim_fun
        elif self.ff_code == 33:
            self.ff_name = "Three-Hump Camel"
            self.bound_min = [-5, -5]
            self.bound_max = [5, 5]
        elif self.ff_code == 34:
            self.ff_name = "Beale"
            self.bound_min = [-4.5, -4.5]
            self.bound_max = [4.5, 4.5]
        elif self.ff_code == 35:
            self.ff_name = "Branin"
            self.bound_min = [-5, 0]
            self.bound_max = [0, 15]
        elif self.ff_code == 36:
            self.ff_name = "Michalewicz"
            self.bound_min = [0] * self.dim_of_multidim_fun
            self.bound_max = [math.pi] * self.dim_of_multidim_fun
        elif self.ff_code == 37:
            self.ff_name = "Colville"
            self.bound_min = [-10, -10, -10, -10]
            self.bound_max = [10, 10, 10, 10]
        return

    def upgrade_vel_swarm(self):
        for i in range(self.size):
            self.population[i].upgrade_vel(self.g_best)
        return

    def upgrade_position_swarm(self):
        for i in range(self.size):
            self.population[i].upgrade_position()
        return

    def upgrade_gbest(self):
        g_best_upgraded = 0
        for i in range(self.size):
            if self.population[i].fitness_value_p_best < self.gbest_value:
                self.gbest_value = self.population[i].fitness_value_p_best
                self.g_best = self.population[i].x
                self.g_best_float_position = self.population[i].float_position
                g_best_upgraded = 1
        return g_best_upgraded

    def upgrade_pbest_swarm(self):
        for i in range(self.size):
            self.population[i].upgrade_pbest()
        return

    def evaluate_fitness_swarm(self):
        for i in range(self.size):
            self.population[i].fitness_function()
        return

# ##TEST CODE
#
#
# swarm = Swarm(size=30, ff_code=1, number_of_decimals=2)
#
# swarm.evaluate_fitness_swarm()
# swarm.upgrade_pbest_swarm()
# swarm.upgrade_gbest()
# print(swarm)
# swarm.upgrade_vel_swarm()
# swarm.upgrade_position_swarm()
