class Parameters:
    def __init__(self, swarm_size, ff_code, number_of_decimals, omega_generator, c1_generator, c2_generator,
                 mutation_mode, v_mutation_factor, algorithm_iterations, dim_of_multidim_fun, dynamic_mutation_mode, sp,
                 ep,
                 dynamic_omega_mode, omega_sp, omega_ep, parameters_selection_mode, parallel_mode, number_of_processes,
                 execute_iterations,
                 accuracy_threshold_per_dim, electromagnetic_problem_mode, i_amplitude_array, n_x, n_y, d_x, d_y, b_x,
                 b_y,
                 theta0, phi0):

        self.parallel_mode = parallel_mode
        self.execute_iterations = execute_iterations
        self.number_of_processes = number_of_processes
        self.swarm_size = swarm_size
        self.ff_code = ff_code
        self.number_of_decimals = number_of_decimals

        self.omega_generator = omega_generator
        self.c1_generator = c1_generator
        self.c2_generator = c2_generator

        self.mutation_mode = mutation_mode
        self.v_mutation_factor = v_mutation_factor

        self.algorithm_iterations = algorithm_iterations

        self.dim_of_multidim_fun = dim_of_multidim_fun

        self.dynamic_mutation_mode = dynamic_mutation_mode
        self.sp = sp
        self.ep = ep

        self.dynamic_omega_mode = dynamic_omega_mode
        self.omega_sp = omega_sp
        self.omega_ep = omega_ep

        self.parameters_selection_mode = parameters_selection_mode
        self.accuracy_threshold_per_dim = accuracy_threshold_per_dim

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

        # Automatically adjust settings based on problem requirements or invalid combinations
        if self.electromagnetic_problem_mode == 1:
            # Adjust settings for electromagnetic problems
            self.number_of_decimals = 0
            self.ff_code = 0

        if self.execute_iterations <= 1:
            # Ensure a minimum number of iterations for statistical analysis
            self.execute_iterations = 2
            print("The algorithm should be executed more than 1 time in order to perform statistical analysis\n"
                  "The execute_iterations variable has been automatically set to 2")

        if self.parameters_selection_mode == 1 and self.electromagnetic_problem_mode == 1:
            # Reset parameters selection mode if invalid combination detected
            self.parameters_selection_mode = 0
            print("Unacceptable combination of modes\n"
                  "The parameter_selection_mode has been automatically set to 0(deactivated) ")

        if self.mutation_mode == 0 and self.dynamic_mutation_mode != 0:
            # Reset dynamic mutation mode if mutation is off
            self.dynamic_mutation_mode = 0
            print("Unacceptable combination of modes\n"
                  "Dynamic mutation is on while mutation is off\n"
                  "The dynamic_mutation_mode has been automatically set to 0(deactivated) ")

        if not all(0 <= value <= 1 for value in
                   [self.sp, self.ep, self.omega_sp, self.omega_ep, self.v_mutation_factor, self.omega_generator,
                    self.c1_generator, self.c2_generator]):
            # Warn if parameter values are outside the expected range
            print("Warning: Some parameter values representing possibilities are outside the [0, 1] range.")

    # Method to update parameters dynamically
    def update_parameters(self, ff_code, swarm_size, v_mutation_factor, sp, ep, omega_generator,
                          omega_sp,
                          omega_ep, c1_generator, c2_generator):
        # Update parameters with new values
        self.ff_code = ff_code
        self.swarm_size = swarm_size
        self.v_mutation_factor = v_mutation_factor
        self.sp = sp
        self.ep = ep
        self.omega_generator = omega_generator
        self.omega_sp = omega_sp
        self.omega_ep = omega_ep
        self.c1_generator = c1_generator
        self.c2_generator = c2_generator

        return

    def param_dict(self):
        # Initialize dictionary with basic parameters

        parameters_dict = {'swarm_size': self.swarm_size}

        # Include mutation parameters if mutation mode is activated
        if self.mutation_mode == 1:
            if self.dynamic_mutation_mode == 0:
                # Add static mutation factor
                parameters_dict.update({
                    'v_mutation_factor': self.v_mutation_factor
                })
            else:
                # Add dynamic mutation parameters
                parameters_dict.update({
                    'sp': self.sp,
                    'ep': self.ep
                })

        # Include omega parameters
        if self.dynamic_omega_mode == 0:
            # Add static omega generator
            parameters_dict.update({
                'omega_generator': self.omega_generator
            })
        else:
            # Add dynamic omega parameters
            parameters_dict.update({
                'omega_sp': self.omega_sp,
                'omega_ep': self.omega_ep
            })

        # Include cognitive and social coefficients
        parameters_dict.update({
            'c1_generator': self.c1_generator,
            'c2_generator': self.c2_generator
        })

        # Round float values to 2 decimal places
        parameters_dict = {
            key: round(value, 2) if isinstance(value, float) else value
            for key, value in parameters_dict.items()
        }
        return parameters_dict

    def factors_sets(self):
        # Create list to store parameter sets

        # Case 1: Static parameters for electromagnetic problem mode
        if self.parameters_selection_mode == 0 and self.electromagnetic_problem_mode == 1:
            factors = [
                (self.ff_code, self.swarm_size, self.v_mutation_factor, self.sp, self.ep,
                 self.omega_generator, self.omega_sp,
                 self.omega_ep, self.c1_generator,
                 self.c2_generator)
            ]
        # Case 2: Parameters for testing a specific version of the algorithm on all test functions
        elif self.parameters_selection_mode == 0 and self.electromagnetic_problem_mode == 0:
            factors = [
                (i_ff_code, self.swarm_size, self.v_mutation_factor, self.sp, self.ep,
                 self.omega_generator, self.omega_sp,
                 self.omega_ep, self.c1_generator,
                 self.c2_generator)
                for i_ff_code in range(1, 48)
            ]
        # Case 3: Parameter sets with mutation mode off and static omega mode
        elif self.parameters_selection_mode == 1 and self.mutation_mode == 0 and self.dynamic_omega_mode == 0:
            factors = [
                (self.ff_code, 10 * i_swarm_size, self.v_mutation_factor, self.sp, self.ep,
                 0.1 * i_omega_generator, self.omega_sp,
                 self.omega_ep, 0.25 * i_c1_generator,
                 0.25 * i_c2_generator)
                for i_swarm_size in range(3, 4)
                for i_omega_generator in range(1, 11)
                for i_c1_generator in range(1, 4)
                for i_c2_generator in range(1, 4)]
        # Case 4: Parameter sets with mutation mode off and dynamic omega
        elif self.parameters_selection_mode == 1 and self.mutation_mode == 0 and self.dynamic_omega_mode != 0:
            factors = [(self.ff_code, 10 * i_swarm_size, self.v_mutation_factor, self.sp, self.ep,
                        self.omega_generator,
                        0.5 + 0.1 * i_omega_sp,
                        0.1 * i_omega_ep,
                        0.25 * i_c1_generator,
                        0.25 * i_c2_generator)
                       for i_swarm_size in range(3, 4)
                       for i_omega_sp in range(1, 5)
                       for i_omega_ep in range(1, 5)
                       for i_c1_generator in range(1, 4)
                       for i_c2_generator in range(1, 4)]
        # Case 5: Parameter sets with static mutation and static omega
        elif self.parameters_selection_mode == 1 and self.dynamic_mutation_mode == 0 and self.dynamic_omega_mode == 0:
            factors = [
                (self.ff_code, 10 * i_swarm_size, 0.1 * i_v_mutation_factor, self.sp, self.ep,
                 0.1 * i_omega_generator, self.omega_sp,
                 self.omega_ep, 0.25 * i_c1_generator,
                 0.25 * i_c2_generator)
                for i_swarm_size in range(3, 4)
                for i_v_mutation_factor in range(1, 11)
                for i_omega_generator in range(1, 11)
                for i_c1_generator in range(1, 4)
                for i_c2_generator in range(1, 4)]
        # Case 6: Parameter sets with dynamic mutation and static omega
        elif self.parameters_selection_mode == 1 and self.dynamic_mutation_mode != 0 and self.dynamic_omega_mode == 0:
            factors = [
                (self.ff_code, 10 * i_swarm_size, self.v_mutation_factor, 0.1 * i_sp, 0.5 + 0.1 * i_ep,
                 0.2 * i_omega_generator,
                 self.omega_sp, self.omega_ep, 0.2 * i_c1_generator,
                 0.2 * i_c2_generator)
                for i_swarm_size in range(3, 4)
                for i_sp in range(1, 5)
                for i_ep in range(1, 5)
                for i_omega_generator in range(1, 5)
                for i_c1_generator in range(1, 5)
                for i_c2_generator in range(1, 5)]
        # Case 7: Parameter sets with static mutation and dynamic omega
        elif self.parameters_selection_mode == 1 and self.dynamic_mutation_mode == 0 and self.dynamic_omega_mode != 0:
            factors = [(self.ff_code, 10 * i_swarm_size, 0.1 * i_v_mutation_factor, self.sp, self.ep,
                        self.omega_generator,
                        0.5 + 0.1 * i_omega_sp,
                        0.1 * i_omega_ep,
                        0.25 * i_c1_generator,
                        0.25 * i_c2_generator)
                       for i_swarm_size in range(3, 4)
                       for i_v_mutation_factor in range(1, 10)
                       for i_omega_sp in range(1, 5)
                       for i_omega_ep in range(1, 5)
                       for i_c1_generator in range(1, 4)
                       for i_c2_generator in range(1, 4)]
        # Case 8: Parameter sets with dynamic mutation and dynamic omega
        elif self.parameters_selection_mode == 1 and self.dynamic_mutation_mode != 0 and self.dynamic_omega_mode != 0:
            factors = [
                (self.ff_code, 10 * i_swarm_size, self.v_mutation_factor, 0.1 * i_sp, 0.5 + 0.1 * i_ep,
                 self.omega_generator,
                 0.5 + 0.1 * i_omega_sp, 0.1 * i_omega_ep,
                 0.25 * i_c_generator,
                 0.25 * i_c_generator)
                for i_swarm_size in range(3, 4)
                for i_sp in range(1, 5)
                for i_ep in range(1, 5)
                for i_omega_sp in range(1, 5)
                for i_omega_ep in range(1, 5)
                for i_c_generator in range(1, 4)]

        return factors

    def print_infos(self):
        # Initialize an empty message string
        message = ""

        # Check parameter selection mode and electromagnetic problem mode
        if self.parameters_selection_mode == 1:
            message += f"Different sets of parameters will be used to test their effectiveness in a particular test function (ff_code= {self.ff_code})\n"
        elif self.parameters_selection_mode == 0 and self.electromagnetic_problem_mode == 0:
            message += f"A specific set of parameters will be used for the optimization of test functions\n" \
                       f"{self.param_dict()}\n"
        elif self.parameters_selection_mode == 0 and self.electromagnetic_problem_mode == 1:
            message += f"A specific set of parameters will be used for the optimization of an array-thinning problem {self.n_x} X {self.n_y}\n" \
                       f"{self.param_dict()}\n"

        # Check mutation mode and dynamic mutation mode
        if self.mutation_mode == 1 and self.dynamic_mutation_mode != 0:
            message += f"The parameter of mutation will be dynamic( dynamic_mutation_mode={self.dynamic_mutation_mode}). So it will change during the iterations\n"
        elif self.mutation_mode == 1 and self.dynamic_mutation_mode == 0:
            message += "The parameter of mutation will be stable\n"
        elif self.mutation_mode == 0:
            message += "Mutation is disabled"

        # Check dynamic omega mode
        if self.dynamic_omega_mode == 0:
            message += "The parameter of omega will be stable\n"
        else:
            message += f"The parameter of inertia-omega will be dynamic( dynamic_omega_mode={self.dynamic_omega_mode}). So it will change during the iterations\n"

        # Print the message
        print(message)
        return
