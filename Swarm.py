from Particle import Particle
import numpy as np


class Swarm:
    def __init__(self, size, ff_code, number_of_decimals, omega_generator, c1_generator, c2_generator,mutation_mode, v_ones_max_percentage):

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

        self.choose_ff()

        for i in range(self.size):
            p = Particle(i, self.ff_code, self.bound_min, self.bound_max, self.number_of_decimals, self.omega_generator,
                         self.c1_generator, self.c2_generator,self.mutation_mode,self.v_ones_max_percentage)
            self.population.append(p)

    def __str__(self):
        for i in range(self.size):
            print(self.population[i])
        print(f"gbest: {self.g_best.bin}\tgbest_fitness_value :{self.gbest_value}")
        return "-------------------------------------------------------------------------------------------------------------"

    def choose_ff(self):
        if self.ff_code == 1:
            self.ff_name = "Eggholder"
            self.bound_min = [-512, -512]
            self.bound_max = [512, 512]
        elif self.ff_code == 2:
            self.ff_name = "Rastring"
            self.bound_min = [-5.12, -5.12]
            self.bound_max = [5.12, 5.12]

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
        for i in range(self.size):
            if self.population[i].fitness_value_p_best < self.gbest_value:
                self.gbest_value = self.population[i].fitness_value_p_best
                self.g_best = self.population[i].x
                self.g_best_float_position = self.population[i].float_position
        return

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
