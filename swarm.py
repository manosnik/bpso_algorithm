from particle import Particle
import numpy as np
import math
from multiprocessing import Pool

# If ff_in_parallel is true AND we deal with the electromagnetic problem software uses more CPUs in order to parallelize
# the computation of fitness function value for swarm's particles.
ff_in_parallel = True


class Swarm:
    def __init__(self, size, ff_code, number_of_decimals, omega_generator, c1_generator, c2_generator, mutation_mode,
                 v_mutation_factor, dim_of_multidim_fun, electromagnetic_problem_mode, i_amplitude_array, n_x,
                 n_y, d_x, d_y, b_x, b_y, theta0, phi0):

        # Initialize Swarm attributes

        self.size = size
        self.ff_code = ff_code
        self.number_of_decimals = number_of_decimals

        self.omega_generator = omega_generator
        self.c1_generator = c1_generator
        self.c2_generator = c2_generator

        self.mutation_mode = mutation_mode
        self.v_mutation_factor = v_mutation_factor

        self.population = []  # List to store the particles in the swarm
        self.gbest_value = np.inf  # Global best fitness value

        self.dim_of_multidim_fun = dim_of_multidim_fun

        self.global_min = None  # Global minimum position
        self.global_min_value = None  # Global minimum fitness value
        self.bound_min = None  # Lower bounds for variables
        self.bound_max = None  # Upper bounds for variables

        self.electromagnetic_problem_mode = electromagnetic_problem_mode
        self.i_amplitude_array = i_amplitude_array
        self.n_x = n_x
        self.n_y = n_y
        self.d_x = d_x
        self.d_y = d_y
        self.b_x = b_x
        self.b_y = b_y
        self.theta0 = theta0
        self.phi0 = phi0

        # Choose the fitness function based on the provided code
        self.choose_ff()

        # Create the swarm population

        for i in range(self.size):
            p = Particle(i, self.ff_code, self.bound_min, self.bound_max, self.number_of_decimals, self.omega_generator,
                         self.c1_generator, self.c2_generator, self.mutation_mode, self.v_mutation_factor,
                         electromagnetic_problem_mode=self.electromagnetic_problem_mode,
                         i_amplitude_array=self.i_amplitude_array
                         , n_x=
                         self.n_x, n_y=self.n_y, d_x=self.d_x, d_y=self.d_y, b_x=self.b_x, b_y=self.b_y,
                         theta0=self.theta0, phi0=self.phi0, ff_in_parallel=ff_in_parallel)
            self.population.append(p)

    def __str__(self):
        # Define the string representation of the Swarm object.
        # Prints the details of each particle in the swarm and the global best information.

        for i in range(self.size):
            print(self.population[i])
        print(
            f"gbest: {self.g_best.bin}\t g_best_float : {self.g_best_float_position}\tgbest_fitness_value :{self.gbest_value}")
        return "---------------------------------------------------------------------------------------------------- "

    def choose_ff(self):
        # Initialize variables based on the selected fitness function code.
        if self.ff_code == 0:
            self.ff_name = f"Electromagnetic Problem planar array antenna {self.n_x}X{self.n_y}"
            self.bound_min = [0] * self.n_x * self.n_y  # Set minimum bounds based on antenna dimensions
            self.bound_max = [1] * self.n_x * self.n_y  # Set maximum bounds based on antenna dimensions
            self.global_min = None
            self.global_min_value = None
        elif self.ff_code == 1:
            self.ff_name = "Ackley"
            self.bound_min = [-32.768] * self.dim_of_multidim_fun
            self.bound_max = [32.768] * self.dim_of_multidim_fun
            self.global_min = [[0] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        elif self.ff_code == 2:
            self.ff_name = "Beale"
            self.bound_min = [-4.5, -4.5]
            self.bound_max = [4.5, 4.5]
            self.global_min = [[3, 0.5]]
            self.global_min_value = 0
        elif self.ff_code == 3:
            self.ff_name = "Bohachevsky"
            self.bound_min = [-100, -100]
            self.bound_max = [100, 100]
            self.global_min = [[0, 0]]
            self.global_min_value = 0
        elif self.ff_code == 4:
            self.ff_name = "Booth"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
            self.global_min = [[1, 3]]
            self.global_min_value = 0
        elif self.ff_code == 5:
            self.ff_name = "Branin"
            self.bound_min = [-5, 0]
            self.bound_max = [0, 15]
            self.global_min = [[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]]
            self.global_min_value = 0.397887
        elif self.ff_code == 6:
            self.ff_name = "Bukin N.6"
            self.bound_min = [-15, -3]
            self.bound_max = [-5, 3]
            self.global_min = [[-10, 1]]
            self.global_min_value = 0
        elif self.ff_code == 7:
            self.ff_name = "Colville"
            self.bound_min = [-10, -10, -10, -10]
            self.bound_max = [10, 10, 10, 10]
            self.global_min = [[1, 1, 1, 1]]
            self.global_min_value = 0
        elif self.ff_code == 8:
            self.ff_name = "Cross-in-Tray"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
            self.global_min = [[1.3491, -1.3491], [1.3491, 1.3491], [-1.3491, 1.3491], [-1.3491, -1.3491]]
            self.global_min_value = -2.06261
        elif self.ff_code == 9:
            self.ff_name = "De Jong N.5"
            self.bound_min = [-65.536, -65.536]
            self.bound_max = [65.536, 65.536]
            self.global_min = None
            self.global_min_value = None
        elif self.ff_code == 10:
            self.ff_name = "Dixon-Price"
            self.bound_min = [-10] * self.dim_of_multidim_fun
            self.bound_max = [10] * self.dim_of_multidim_fun
            self.global_min = [[]]
            for i in range(self.dim_of_multidim_fun):
                self.global_min[0].append(2 ** (-(2 ** (i + 1) - 2) / 2 ** (i + 1)))
            self.global_min_value = 0
        elif self.ff_code == 11:
            self.ff_name = "Dropwave"
            self.bound_min = [-5.12, -5.12]
            self.bound_max = [5.12, 5.12]
            self.global_min = [[0, 0]]
            self.global_min_value = -1
        elif self.ff_code == 12:
            self.ff_name = "Easom"
            self.bound_min = [-100, -100]
            self.bound_max = [100, 100]
            self.global_min = [[math.pi, math.pi]]
            self.global_min_value = -1
        elif self.ff_code == 13:
            self.ff_name = "Eggholder"
            self.bound_min = [-512, -512]
            self.bound_max = [512, 512]
            self.global_min = [[512, 404.2319]]
            self.global_min_value = -959.6407
        elif self.ff_code == 14:
            self.ff_name = "Forrester"
            self.bound_min = [0]
            self.bound_max = [1]
            self.global_min = None
            self.global_min_value = None
        elif self.ff_code == 15:
            self.ff_name = "Goldstein-Price"
            self.bound_min = [-2, -2]
            self.bound_max = [2, 2]
            self.global_min = [[0, -1]]
            self.global_min_value = 3
        elif self.ff_code == 16:
            self.ff_name = "Gramacy & Lee Function"
            self.bound_min = [0.5]
            self.bound_max = [2.5]
            self.global_min = None
            self.global_min_value = None
        elif self.ff_code == 17:
            self.ff_name = "Griewank"
            self.bound_min = [-600] * self.dim_of_multidim_fun
            self.bound_max = [600] * self.dim_of_multidim_fun
            self.global_min = [[0] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        elif self.ff_code == 18:
            self.ff_name = "Hartmann 3-D"
            self.bound_min = [0, 0, 0]
            self.bound_max = [1, 1, 1]
            self.global_min = [[0.114614, 0.555649, 0.852547]]
            self.global_min_value = -3.86278
        elif self.ff_code == 19:
            self.ff_name = "Hartmann 4-D"
            self.bound_min = [0, 0, 0, 0]
            self.bound_max = [1, 1, 1, 1]
            self.global_min = [[0.1873, 0.1906, 0.5566, 0.2647]]
            self.global_min_value = -3.135474
        elif self.ff_code == 20:
            self.ff_name = "Hartmann 6-D"
            self.bound_min = [0, 0, 0, 0, 0, 0]
            self.bound_max = [1, 1, 1, 1, 1, 1]
            self.global_min = [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]
            self.global_min_value = -3.32237
        elif self.ff_code == 21:
            self.ff_name = "Holder Table"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
            self.global_min = [[8.05502, 9.66459], [8.05502, -9.66459], [-8.05502, 9.66459], [-8.05502, -9.66459]]
            self.global_min_value = -19.2085
        elif self.ff_code == 22:
            self.ff_name = "Langermann"
            self.bound_min = [0, 0]
            self.bound_max = [10, 10]
            self.global_min = None
            self.global_min_value = None
        elif self.ff_code == 23:
            self.ff_name = "Levy"
            self.bound_min = [-10] * self.dim_of_multidim_fun
            self.bound_max = [10] * self.dim_of_multidim_fun
            self.global_min = [[1] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        elif self.ff_code == 24:
            self.ff_name = "Levy N.13"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
            self.global_min = [[1, 1]]
            self.global_min_value = 0
        elif self.ff_code == 25:
            self.ff_name = "Matyas"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
            self.global_min = [[0, 0]]
            self.global_min_value = 0
        elif self.ff_code == 26:
            self.ff_name = "McCormick"
            self.bound_min = [-1.5, -3]
            self.bound_max = [4, 4]
            self.global_min = [[-0.54719, -1.54719]]
            self.global_min_value = -1.9133
        elif self.ff_code == 27:
            self.ff_name = "Michalewicz"
            self.bound_min = [0] * self.dim_of_multidim_fun
            self.bound_max = [math.pi] * self.dim_of_multidim_fun
            if self.dim_of_multidim_fun == 2:
                self.global_min = [[2.2, 1.57]]
                self.global_min_value = -1.8013
            elif self.dim_of_multidim_fun == 5:
                self.global_min = None
                self.global_min_value = -4.687658
            elif self.dim_of_multidim_fun == 10:
                self.global_min = None
                self.global_min_value = -9.66015
            else:
                self.global_min = None
                self.global_min_value = None
        elif self.ff_code == 28:
            self.ff_name = "Perm 0,d,b"
            self.bound_min = [-self.dim_of_multidim_fun] * self.dim_of_multidim_fun
            self.bound_max = [self.dim_of_multidim_fun] * self.dim_of_multidim_fun
            self.global_min = [[]]
            for i in range(self.dim_of_multidim_fun):
                self.global_min[0].append(1 / (i + 1))
            self.global_min_value = 0
        elif self.ff_code == 29:
            self.ff_name = "Perm d,b"
            self.bound_min = [-self.dim_of_multidim_fun] * self.dim_of_multidim_fun
            self.bound_max = [self.dim_of_multidim_fun] * self.dim_of_multidim_fun
            self.global_min = [[]]
            for i in range(self.dim_of_multidim_fun):
                self.global_min[0].append(i + 1)
            self.global_min_value = 0
        elif self.ff_code == 30:
            self.ff_name = "Powell"
            self.dim_of_multidim_fun = math.ceil(self.dim_of_multidim_fun / 4) * 4
            self.bound_min = [-4] * self.dim_of_multidim_fun
            self.bound_max = [5] * self.dim_of_multidim_fun
            self.global_min = [[0] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        elif self.ff_code == 31:
            self.ff_name = "Power Sum function"
            self.bound_min = [0, 0, 0, 0]
            self.bound_max = [4, 4, 4, 4]
            self.global_min = None
            self.global_min_value = None
        elif self.ff_code == 32:
            self.ff_name = "Rastring"
            self.bound_min = [-5.12] * self.dim_of_multidim_fun
            self.bound_max = [5.12] * self.dim_of_multidim_fun
            self.global_min = [[0] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        elif self.ff_code == 33:
            self.ff_name = "Rosenbrock"
            self.bound_min = [-5] * self.dim_of_multidim_fun
            self.bound_max = [10] * self.dim_of_multidim_fun
            self.global_min = [[1] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        elif self.ff_code == 34:
            self.ff_name = "Rotated Hyper-Ellipsoid"
            self.bound_min = [-65.536] * self.dim_of_multidim_fun
            self.bound_max = [65.536] * self.dim_of_multidim_fun
            self.global_min = [[0] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        elif self.ff_code == 35:
            self.ff_name = "Schaffer N.2"
            self.bound_min = [-100, -100]
            self.bound_max = [100, 100]
            self.global_min = [[0, 0]]
            self.global_min_value = 0
        elif self.ff_code == 36:
            self.ff_name = "Schaffer N.4"
            self.bound_min = [-100, -100]
            self.bound_max = [100, 100]
            self.global_min = None
            self.global_min_value = None
        elif self.ff_code == 37:
            self.ff_name = "Schwefel"
            self.bound_min = [-500] * self.dim_of_multidim_fun
            self.bound_max = [500] * self.dim_of_multidim_fun
            self.global_min = [[420.9687] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        elif self.ff_code == 38:
            self.ff_name = "Shekel"
            self.bound_min = [0, 0, 0, 0]
            self.bound_max = [10, 10, 10, 10]
            self.global_min = [[4, 4, 4, 4]]
            self.global_min_value = -10.5364
        elif self.ff_code == 39:
            self.ff_name = "Shubert"
            self.bound_min = [-10, -10]
            self.bound_max = [10, 10]
            self.global_min = None
            self.global_min_value = -186.7309
        elif self.ff_code == 40:
            self.ff_name = "Six-Hump Camel"
            self.bound_min = [-3, -2]
            self.bound_max = [3, 2]
            self.global_min = [[0.0898, -0.7126], [-0.0898, 0.7126]]
            self.global_min_value = -1.0316
        elif self.ff_code == 41:
            self.ff_name = "Sphere"
            self.bound_min = [-5.12] * self.dim_of_multidim_fun
            self.bound_max = [5.12] * self.dim_of_multidim_fun
            self.global_min = [[0] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        elif self.ff_code == 42:
            self.ff_name = "Styblinski-Tang"
            self.bound_min = [-5] * self.dim_of_multidim_fun
            self.bound_max = [5] * self.dim_of_multidim_fun
            self.global_min = [[-2.903534] * self.dim_of_multidim_fun]
            self.global_min_value = -39.16599 * self.dim_of_multidim_fun
        elif self.ff_code == 43:
            self.ff_name = "Sum Squares function"
            self.bound_min = [-10] * self.dim_of_multidim_fun
            self.bound_max = [10] * self.dim_of_multidim_fun
            self.global_min = [[0] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        elif self.ff_code == 44:
            self.ff_name = "Sum of Different Powers function"
            self.bound_min = [-1] * self.dim_of_multidim_fun
            self.bound_max = [1] * self.dim_of_multidim_fun
            self.global_min = [[0] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        elif self.ff_code == 45:
            self.ff_name = "Three-Hump Camel"
            self.bound_min = [-5, -5]
            self.bound_max = [5, 5]
            self.global_min = [[0, 0]]
            self.global_min_value = 0
        elif self.ff_code == 46:
            self.ff_name = "Trid"
            self.bound_min = [-(self.dim_of_multidim_fun ** 2)] * self.dim_of_multidim_fun
            self.bound_max = [self.dim_of_multidim_fun ** 2] * self.dim_of_multidim_fun
            self.global_min = [[]]
            for i in range(self.dim_of_multidim_fun):
                self.global_min[0].append((i + 1) * (self.dim_of_multidim_fun - i))
            self.global_min_value = -self.dim_of_multidim_fun * (self.dim_of_multidim_fun + 4) * (
                    self.dim_of_multidim_fun - 1) / 6
        elif self.ff_code == 47:
            self.ff_name = "Zakharov"
            self.bound_min = [-5] * self.dim_of_multidim_fun
            self.bound_max = [10] * self.dim_of_multidim_fun
            self.global_min = [[0] * self.dim_of_multidim_fun]
            self.global_min_value = 0
        return

    def upgrade_vel_swarm(self):
        # Upgrade velocity for each particle in the swarm.
        for i in range(self.size):
            # Set mutation factor and generators
            self.population[i].v_mutation_factor = self.v_mutation_factor
            self.population[i].omega_generator = self.omega_generator
            # Call the particle's upgrade velocity method
            self.population[i].upgrade_vel(self.g_best)
        return

    def upgrade_position_swarm(self):
        # Upgrade position for each particle in the swarm.
        for i in range(self.size):
            # Call the particle's upgrade position method
            self.population[i].upgrade_position()
        return

    def upgrade_gbest(self):
        # Upgrade the global best particle among the swarm.
        g_best_upgraded = 0
        for i in range(self.size):
            if self.population[i].fitness_value_p_best < self.gbest_value:
                # Update global best value, position, and float position
                self.gbest_value = self.population[i].fitness_value_p_best
                self.g_best = self.population[i].x
                self.g_best_float_position = self.population[i].float_position
                g_best_upgraded = 1
        return g_best_upgraded

    def upgrade_pbest_swarm(self):
        # Upgrade personal best for each particle in the swarm.
        for i in range(self.size):
            # Call the particle's upgrade personal best method
            self.population[i].upgrade_pbest()
        return

    # Upgrade the value of the fitness function for every particle in the Swarm .
    # For test functions it happens in serial.
    # If we are in electromagnetic problem mode we can choose a parallel way to speed up the procedure
    def evaluate_fitness_swarm(self):
        # Evaluate fitness for each particle in the swarm.
        if ff_in_parallel == False or self.electromagnetic_problem_mode == 0:
            # Serial evaluation for non-parallelizable cases or test functions
            for i in range(self.size):
                self.population[i].fitness_function()
        elif ff_in_parallel == True and self.electromagnetic_problem_mode == 1:
            # Parallel evaluation for electromagnetic problem mode
            # Split the particles' fitness value calculation to more processes for parallelization
            with Pool(processes=2) as pool:  # Change 'processes' value to split the workload across more processes
                a = pool.map(Particle.fitness_function, self.population)
                pool.close()
                pool.join()
            for i in range(self.size):
                # Assign fitness values obtained from parallel evaluation
                self.population[i].fitness_value = a[i]
        return

# ##TEST CODE

#
#
# swarm = Swarm(size=30, ff_code=0, number_of_decimals=0,omega_generator=0.6,c1_generator=0.4,c2_generator=0.4,
# mutation_mode=1,v_mutation_factor=0.2,dim_of_multidim_fun=4)
#
# for i in range(10):
#     swarm.evaluate_fitness_swarm()
#     swarm.upgrade_pbest_swarm()
#     swarm.upgrade_gbest()
#     print(swarm)
#     swarm.upgrade_vel_swarm()
#     swarm.upgrade_position_swarm()
