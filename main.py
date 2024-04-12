import numpy as np
import os
import support_functions as sup
from timeit import default_timer as timer
from results import Results
from parameters import Parameters
from diagrams import diagrams

# Disable logging for Ray to avoid cluttering the output
os.environ["RAY_DEDUP_LOGS"] = "0"  # Set environment variable to disable Ray logs
import ray  # Import ray module for parallel processing

######  MANUAL  INPUT ##########

execute_iterations = 4  # Number of times the optimization algorithm will be executed
algorithm_iterations = 1000  # Number of iterations for the optimization algorithm
number_of_decimals = 4  # Number of decimal places used for encoding and decoding the solutions
dim_of_multidim_fun = 5  # Dimensionality of the multimodal function

mutation_mode = 1  # Mode for mutation in the algorithm
dynamic_mutation_mode = 1  # Mode for dynamic mutation in the algorithm ---1->linear---2->exponential----3->oscillation
sp = 0.1  # Start probability for dynamic mutation
ep = 0.8  # End probability for dynamic mutation
dynamic_omega_mode = 1  # Mode for dynamic omega in the algorithm ---1->linear
omega_sp = 0.9  # Start value for dynamic omega
omega_ep = 0.4  # End value for dynamic omega

parallel_mode = 0  # Mode for parallel processing
number_of_processes = 4  # Number of processes used for parallel processing

parameters_selection_mode = 1  # Mode for selecting parameters
accuracy_threshold_per_dim = 0.5  # Accuracy threshold per dimension
ff_code = 11  # Fitness function code
swarm_size = 30  # Size of the swarm

v_mutation_factor = 0.3  # Mutation factor for velocity update
omega_generator = 0.8  # Generator for omega (inertia coefficient)
c1_generator = 0.5  # Generator for c1 (cognitive coefficient)
c2_generator = 0.5  # Generator for c2 (social coefficient)

electromagnetic_problem_mode = 1  # Mode for electromagnetic problem
n_x = 4  # Number of elements in x-direction for planar array antenna
n_y = 8  # Number of elements in y-direction for planar array antenna
i_amplitude_array = np.array([1] * n_x * n_y, dtype=np.int32)  # Amplitude array for planar array antenna
d_x = 1 / 2  # Distance between elements in x-direction
d_y = 1 / 2  # Distance between elements in y-direction
# b_x = 0  # Angle b_x (rad) for planar array antenna
# b_y = 0  # Angle b_y (rad) for planar array antenna
# theta0 = 0  # Initial theta angle (rad) for planar array antenna
# phi0 = 0  # Initial phi angle (rad) for planar array antenna

# Uncomment to use alternative values for b_x, b_y, theta0, and phi0
b_x = -(3 * np.pi) / 4
b_y = -(np.sqrt(3) * np.pi) / 4
theta0 = np.pi / 3
phi0 = np.pi / 6

###############################

# ff_name_array contains names of fitness functions
ff_name_array = ["Electromagnetic Problem", "Ackley (a=20 , b=0.2 , c=2*pi) Dim:  " + str(dim_of_multidim_fun),
                 "Beale", "Bohachevsky",
                 "Booth", "Branin", "Bukin N.6", "Colville", "Cross-in-Tray", "De Jong N.5",
                 "Dixon-Price Dim:  " + str(dim_of_multidim_fun),
                 "Dropwave", "Easom", "Eggholder", "Forrester", "Goldstein-Price", "Gramacy - Lee Function",
                 "Griewank Dim:  " + str(dim_of_multidim_fun), "Hartmann 3-D", "Hartmann 4-D", "Hartmann 6-D",
                 "Holder Table", "Langermann", "Levy Dim:  " + str(dim_of_multidim_fun), "Levy N.13", "Matyas",
                 "McCormick",
                 "Michalewicz Dim:  " + str(dim_of_multidim_fun), "Perm 0,d,b", "Perm d,b", "Powell",
                 "Power Sum Function", "Rastring", "Rosenbrock Dim:  " + str(dim_of_multidim_fun),
                 "Rotated Hyper-Ellipsoid Dim:  " + str(dim_of_multidim_fun), "Schaffer N.2",
                 "Schaffer N.4", "Schwefel Dim:  " + str(dim_of_multidim_fun), "Shekel", "Shubert",
                 "Six-Hump Camel", "Sphere", "Styblinski-Tang",
                 "Sum Squares function Dim:  " + str(dim_of_multidim_fun),
                 "Sum of Different Powers function Dim:  " + str(dim_of_multidim_fun), "Three-Hump Camel",
                 "Trid Dim:  " + str(dim_of_multidim_fun), "Zakharov Dim:  " + str(dim_of_multidim_fun)]

if __name__ == '__main__':

    # Start timing the duration of the optimization process
    duration_start = timer()

    # If parallel mode is enabled, initialize Ray and print version information
    if parallel_mode == 1:
        print(ray.__version__)
        ray.init(num_cpus=number_of_processes)
        print(ray.available_resources(), flush=True)

    # Open a file for writing results
    file = open('Data_file.txt', 'w')

    # Initialize parameters based on provided values
    parameters = Parameters(swarm_size, ff_code, number_of_decimals, omega_generator, c1_generator,
                            c2_generator, mutation_mode, v_mutation_factor, algorithm_iterations, dim_of_multidim_fun,
                            dynamic_mutation_mode, sp, ep,
                            dynamic_omega_mode, omega_sp, omega_ep, parameters_selection_mode, parallel_mode,
                            number_of_processes, execute_iterations,
                            accuracy_threshold_per_dim, electromagnetic_problem_mode, i_amplitude_array, n_x,
                            n_y, d_x, d_y, b_x, b_y, theta0, phi0)

    # Print information about the parameters
    parameters.print_infos()

    # Initialize Results object to store optimization results
    results = Results(parameters)

    # Generate sets of factors for parameter testing
    factors = parameters.factors_sets()

    # Loop through the factors sets and execute optimization
    for ff_code, swarm_size, v_mutation_factor, sp, ep, omega_generator, omega_sp, omega_ep, c1_generator, c2_generator in factors:

        # Update parameters for current factor set
        parameters.update_parameters(ff_code, swarm_size, v_mutation_factor, sp, ep, omega_generator,
                                     omega_sp, omega_ep, c1_generator, c2_generator)

        # Create dictionary of parameters
        param_dictionary = parameters.param_dict()

        # Print terminal information
        sup.terminal_infos(parameters, param_dictionary, ff_name_array)

        # Execute optimization
        results = sup.execution(parameters, results)

        # Compute unknown global minimum if needed
        sup.compute_unknown_global_min(parameters, results)

        # Edit text and data that produced from every optimization
        sup.text_and_data_editor(parameters, results, param_dictionary, ff_name_array, file)

        # Clear arrays in results
        results.clear_arrays()

    # Edit the end of text file
    sup.text_closure_editor(parameters, results, param_dictionary, file)

    # Generate diagrams
    diagrams(parameters, results, param_dictionary, ff_name_array)

    # Print total duration of the optimization process
    sup.print_duration(duration_start)

    # If parallel mode is enabled, shut down Ray
    if parallel_mode == 1:
        ray.shutdown()

    # Close the result file
    file.close()
