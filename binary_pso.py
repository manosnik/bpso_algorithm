from swarm import Swarm
import math
import os

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray


class BinaryPso:
    def __init__(self, size, ff_code, number_of_decimals, omega_generator, c1_generator, c2_generator,
                 mutation_mode, v_mutation_factor, algorithm_iterations, dim_of_multidim_fun, dynamic_mutation_mode, sp,
                 ep, dynamic_omega_mode, omega_sp, omega_ep, electromagnetic_problem_mode, i_amplitude_array, n_x,
                 n_y, d_x, d_y, b_x, b_y, theta0, phi0):

        # Initialize Binary PSO parameters
        self.size = size
        self.ff_code = ff_code
        self.number_of_decimals = number_of_decimals
        self.c1_generator = c1_generator
        self.c2_generator = c2_generator

        self.mutation_mode = mutation_mode
        self.dynamic_mutation_mode = dynamic_mutation_mode
        self.sp = sp
        self.ep = ep

        # Set mutation factor based on dynamic mutation mode
        if self.dynamic_mutation_mode == 0:
            self.v_mutation_factor = v_mutation_factor
        else:
            self.v_mutation_factor = self.sp

        self.dynamic_omega_mode = dynamic_omega_mode
        self.omega_sp = omega_sp
        self.omega_ep = omega_ep

        # Set omega generator based on dynamic omega mode
        if self.dynamic_omega_mode == 0:
            self.omega_generator = omega_generator
        else:
            self.omega_generator = self.omega_sp

        self.algorithm_iterations = algorithm_iterations

        self.dim_of_multidim_fun = dim_of_multidim_fun

        self.iteration_of_gbest = None

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

        # Initialize Swarm
        self.swarm = Swarm(size=self.size, ff_code=self.ff_code, number_of_decimals=self.number_of_decimals,
                           omega_generator=self.omega_generator, c1_generator=self.c1_generator,
                           c2_generator=self.c2_generator, mutation_mode=self.mutation_mode,
                           v_mutation_factor=self.v_mutation_factor, dim_of_multidim_fun=self.dim_of_multidim_fun,
                           electromagnetic_problem_mode=self.electromagnetic_problem_mode,
                           i_amplitude_array=self.i_amplitude_array
                           , n_x=self.n_x, n_y=self.n_y, d_x=self.d_x, d_y=self.d_y, b_x=self.b_x, b_y=self.b_y,
                           theta0=self.theta0, phi0=self.phi0)

    def play(self):
        # Initialize variables for controlling mutation dynamics

        k = 0.005  # Constant for exponential mutation mode 2
        flag = 0  # Flag indication to change mutation value for mutation mode 3
        counter = 0  # counter of iterations with no gbest change for mutation mode 3
        iter_percentage = 0.15  # Percentage of iterations for mutation mode 3

        # Controls the flow of the algorithm
        # Iterate over algorithm iterations
        for i in range(self.algorithm_iterations):
            # Log current iteration if electromagnetic problem mode is enabled
            # Serves for monitoring progress through terminal
            if self.electromagnetic_problem_mode == 1:
                ray.logger.info(f"Algorithm Iteration {i}")
            # Evaluate fitness, update personal and global best
            self.swarm.evaluate_fitness_swarm()
            self.swarm.upgrade_pbest_swarm()
            g_best_upgraded = self.swarm.upgrade_gbest()
            # If global best has been updated saves the iteration number
            if g_best_upgraded == 1:
                self.iteration_of_gbest = i

            # Update velocity and position of particles
            self.swarm.upgrade_vel_swarm()
            self.swarm.upgrade_position_swarm()

            # Control mutation factors based on mutation mode
            if self.mutation_mode == 1:
                if self.dynamic_mutation_mode == 1:   # Linear change in mutation factor
                    self.swarm.v_mutation_factor = self.sp + ((self.ep - self.sp) / self.algorithm_iterations) * i
                elif self.dynamic_mutation_mode == 2:  # Exponential change in mutation factor
                    self.swarm.v_mutation_factor = self.ep + (self.sp - self.ep) * math.exp(-k * i)
                elif self.dynamic_mutation_mode == 3:  # Fluctuating mutation factor
                    if g_best_upgraded == 0:
                        counter += 1
                        if counter > self.algorithm_iterations * iter_percentage:
                            flag = 1
                            counter = 0
                    elif g_best_upgraded == 1:
                        counter = 0
                    if flag == 1 and self.swarm.v_mutation_factor == self.sp:
                        self.swarm.v_mutation_factor = self.ep
                        flag = 0
                    elif flag == 1 and self.swarm.v_mutation_factor == self.ep:
                        self.swarm.v_mutation_factor = self.sp
                        flag = 0
            # Control omega (inertia weight) based on dynamic omega mode
            if self.dynamic_omega_mode == 1:  # Linear change in omega
                self.swarm.omega_generator = self.omega_sp + (
                        (self.omega_ep - self.omega_sp) / self.algorithm_iterations) * i
        return
