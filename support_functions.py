import math
import statistics
from timeit import default_timer as timer

import numpy as np

from binary_pso import BinaryPso
from results import Results
from results import ResultsRay
from parameters import Parameters
from antenna_functions import directivity_planar_array_antenna_c as dir_function
from antenna_functions import sll_planar_array_antenna_c as sll_function
from scipy.spatial import distance
import os

# Disabling Ray deduplication logs
os.environ["RAY_DEDUP_LOGS"] = "0"
# Importing Ray for parallel processing
import ray


# Remote function to run the algorithm in parallel
@ray.remote
def run_alg(parameters):
    # Start timer
    start = timer()
    # Initialize BPSO algorithm
    algorithm = BinaryPso(size=parameters.swarm_size, ff_code=parameters.ff_code,
                          number_of_decimals=parameters.number_of_decimals,
                          omega_generator=parameters.omega_generator, c1_generator=parameters.c1_generator,
                          c2_generator=parameters.c2_generator,
                          mutation_mode=parameters.mutation_mode,
                          v_mutation_factor=parameters.v_mutation_factor,
                          algorithm_iterations=parameters.algorithm_iterations,
                          dim_of_multidim_fun=parameters.dim_of_multidim_fun,
                          dynamic_mutation_mode=parameters.dynamic_mutation_mode, sp=parameters.sp, ep=parameters.ep,
                          dynamic_omega_mode=parameters.dynamic_omega_mode, omega_sp=parameters.omega_sp,
                          omega_ep=parameters.omega_ep,
                          electromagnetic_problem_mode=parameters.electromagnetic_problem_mode,
                          i_amplitude_array=parameters.i_amplitude_array
                          , n_x=parameters.n_x, n_y=parameters.n_y, d_x=parameters.d_x, d_y=parameters.d_y,
                          b_x=parameters.b_x, b_y=parameters.b_y,
                          theta0=parameters.theta0, phi0=parameters.phi0)

    # Run the algorithm
    algorithm.play()

    # Calculate duration
    duration = timer() - start

    # Initialize variables for fitness index, accuracy hit, directivity, and SLL
    fitness_index_contribution = None
    results_accuracy_hit = None
    directivity = None
    sll = None

    # If it's not an electromagnetic problem, calculate fitness index and accuracy hit
    if parameters.electromagnetic_problem_mode == 0:

        # Global min is needed to evaluate our results If we don't , the solution gives the compute_unknown_global_min() function.
        if algorithm.swarm.global_min is not None:
            fitness_index_contribution = min(
                [distance.euclidean(point, algorithm.swarm.g_best_float_position) for point in
                 algorithm.swarm.global_min])
            if fitness_index_contribution < parameters.accuracy_threshold_per_dim * len(algorithm.swarm.bound_min):
                results_accuracy_hit = 1
            else:
                results_accuracy_hit = 0

    # If it's an electromagnetic problem, calculate directivity and SLL
    if parameters.electromagnetic_problem_mode == 1:
        directivity = dir_function(active_elements_array=algorithm.swarm.g_best_float_position,
                                   i_amplitude_array=parameters.i_amplitude_array, d_x=parameters.d_x,
                                   d_y=parameters.d_y,
                                   b_x=parameters.b_x, b_y=parameters.b_y, n_x=parameters.n_x, n_y=parameters.n_y,
                                   theta0=parameters.theta0, phi0=parameters.phi0)

        sll = sll_function(active_elements_array=algorithm.swarm.g_best_float_position,
                           i_amplitude_array=parameters.i_amplitude_array, d_x=parameters.d_x, d_y=parameters.d_y,
                           b_x=parameters.b_x, b_y=parameters.b_y, n_x=parameters.n_x, n_y=parameters.n_y
                           )

    # Create ResultsRay object to store results and return it
    results_ray = ResultsRay(duration=duration, gbest_value=algorithm.swarm.gbest_value,
                             gbest_bin=algorithm.swarm.g_best.bin,
                             gbest_float_position=algorithm.swarm.g_best_float_position,
                             iter_of_gbest=algorithm.iteration_of_gbest, global_min=algorithm.swarm.global_min,
                             global_min_value=algorithm.swarm.global_min_value,
                             parameter_fitness_index=fitness_index_contribution,
                             accuracy_hit=results_accuracy_hit, directivity=directivity,
                             sll=sll)
    return results_ray


# Function for serial execution of the algorithm
def serial_execution(parameters: Parameters, results: Results):
    for i in range(parameters.execute_iterations):
        start = timer()

        # Initialize BPSO algorithm
        algorithm = BinaryPso(size=parameters.swarm_size, ff_code=parameters.ff_code,
                              number_of_decimals=parameters.number_of_decimals,
                              omega_generator=parameters.omega_generator, c1_generator=parameters.c1_generator,
                              c2_generator=parameters.c2_generator,
                              mutation_mode=parameters.mutation_mode,
                              v_mutation_factor=parameters.v_mutation_factor,
                              algorithm_iterations=parameters.algorithm_iterations,
                              dim_of_multidim_fun=parameters.dim_of_multidim_fun,
                              dynamic_mutation_mode=parameters.dynamic_mutation_mode, sp=parameters.sp,
                              ep=parameters.ep,
                              dynamic_omega_mode=parameters.dynamic_omega_mode, omega_sp=parameters.omega_sp,
                              omega_ep=parameters.omega_ep,
                              electromagnetic_problem_mode=parameters.electromagnetic_problem_mode,
                              i_amplitude_array=parameters.i_amplitude_array
                              , n_x=parameters.n_x, n_y=parameters.n_y, d_x=parameters.d_x, d_y=parameters.d_y,
                              b_x=parameters.b_x, b_y=parameters.b_y,
                              theta0=parameters.theta0, phi0=parameters.phi0)

        # Run the algorithm
        algorithm.play()

        # Store results
        results.duration.append(timer() - start)
        results.gbest_value.append(algorithm.swarm.gbest_value)
        results.gbest_bin.append(algorithm.swarm.g_best.bin)
        results.gbest_float_position.append(algorithm.swarm.g_best_float_position)
        results.iter_of_gbest.append(algorithm.iteration_of_gbest)

        # If it's an electromagnetic problem, calculate directivity and SLL and store them
        if parameters.electromagnetic_problem_mode == 1:
            results.directivity.append(dir_function(active_elements_array=algorithm.swarm.g_best_float_position,
                                                    i_amplitude_array=parameters.i_amplitude_array, d_x=parameters.d_x,
                                                    d_y=parameters.d_y, b_x=parameters.b_x, b_y=parameters.b_y,
                                                    n_x=parameters.n_x, n_y=parameters.n_y, theta0=parameters.theta0,
                                                    phi0=parameters.phi0))
            results.sll.append(sll_function(active_elements_array=algorithm.swarm.g_best_float_position,
                                            i_amplitude_array=parameters.i_amplitude_array, d_x=parameters.d_x,
                                            d_y=parameters.d_y, b_x=parameters.b_x, b_y=parameters.b_y,
                                            n_x=parameters.n_x, n_y=parameters.n_y))
        else:
            # If it's not an electromagnetic problem, calculate fitness index and accuracy hit and store them
            fitness_index_contribution = None
            results_accuracy_hit = None

            # Global min is needed to evaluate our results If we don't , the solution gives the compute_unknown_global_min() function.
            if algorithm.swarm.global_min is not None:
                fitness_index_contribution = min(
                    [distance.euclidean(point, algorithm.swarm.g_best_float_position) for point
                     in
                     algorithm.swarm.global_min])
                if fitness_index_contribution < parameters.accuracy_threshold_per_dim * len(
                        algorithm.swarm.bound_min):
                    results_accuracy_hit = 1
                else:
                    results_accuracy_hit = 0

            results.parameter_fitness_index.append(fitness_index_contribution)
            results.accuracy_hit_array.append(results_accuracy_hit)
            # Set the global minimum of the results object to the global_min of the first iteration (as all are the same)
            if i == 0:
                results.global_min = algorithm.swarm.global_min
                results.global_min_value = algorithm.swarm.global_min_value

    return results


# Function for parallel execution of the algorithm
def parallel_execution(parameters: Parameters, results: Results):
    # Convert parameters to Ray object
    input_ray = ray.put(parameters)
    # Run the algorithm remotely in parallel for the specified number of iterations
    results_ray = ray.get([run_alg.remote(input_ray) for _ in range(parameters.execute_iterations)])

    # Process the results obtained from each parallel execution
    for j in range(parameters.execute_iterations):
        # Append the results of each iteration to the total results object
        results.gbest_value.append(results_ray[j].gbest_value)
        results.duration.append(results_ray[j].duration)
        results.gbest_bin.append(results_ray[j].gbest_bin)
        results.gbest_float_position.append(results_ray[j].gbest_float_position)
        results.iter_of_gbest.append(results_ray[j].iter_of_gbest)
        if parameters.electromagnetic_problem_mode == 1:
            results.directivity.append(results_ray[j].directivity)
            results.sll.append(results_ray[j].sll)
        else:
            results.parameter_fitness_index.append(results_ray[j].parameter_fitness_index)
            results.accuracy_hit_array.append(results_ray[j].accuracy_hit)
    # Set the global minimum of the results object to the global_min of the first iteration (as all are the same)
    results.global_min = results_ray[0].global_min
    results.global_min_value = results_ray[0].global_min_value

    return results


# Main execution function
def execution(parameters: Parameters, results: Results):
    if parameters.parallel_mode == 1:
        # Execute in parallel
        results = parallel_execution(parameters, results)

    else:
        # Execute serially
        results = serial_execution(parameters, results)

    return results


# Function to compute the unknown global minimum
# In this case the metrics of the optimization accuracy are computed based on the gbest value and not gbest position
def compute_unknown_global_min(parameters: Parameters, results: Results):
    # Check if it's not an electromagnetic problem - if it is we cannot know global min
    if parameters.electromagnetic_problem_mode == 0:
        # If the global_min is not known
        if results.global_min is None:
            # If the global_min_value is not known
            if results.global_min_value is None:
                # Compute the global_min_value as the minimum gbest_value
                results.global_min_value = round(min(results.gbest_value), 4)
            # Loop through each iteration's results
            for j in range(parameters.execute_iterations):
                # Compute the fitness index metric for each iteration
                results.parameter_fitness_index[j] = np.abs(results.gbest_value[j] - results.global_min_value)
                # Check if the fitness index is within the accuracy threshold and set the hit
                if results.parameter_fitness_index[j] < parameters.accuracy_threshold_per_dim:
                    results.accuracy_hit_array[j] = 1
                else:
                    results.accuracy_hit_array[j] = 0

    return


# Function to edit text and store data in the results file
def text_and_data_editor(parameters: Parameters, results: Results, param_dictionary, ff_name_array, file):
    # Check if it's not an electromagnetic problem
    if parameters.electromagnetic_problem_mode == 0:
        # Write details for non-electromagnetic problems
        file.write(
            f"\n\n{ff_name_array[parameters.ff_code]} (global min ={results.global_min_value} at {results.global_min}) {param_dictionary}\n")
        # Loop through each iteration's results and write details
        for j in range(parameters.execute_iterations):
            file.write(
                f"Execution number {j}\nBest is {results.gbest_value[j]} at {results.gbest_bin[j]} or {results.gbest_float_position[j]} found at iteration {results.iter_of_gbest[j]} _ index= {results.parameter_fitness_index[j]} hit={results.accuracy_hit_array[j]}\n")

        # Compute fitness index sum and accuracy for the function
        ff_parameters_fitness_index = sum(results.parameter_fitness_index)
        ff_parameters_accuracy = (sum(results.accuracy_hit_array) / parameters.execute_iterations) * 100

    else:
        # Write details for electromagnetic problems
        file.write(
            f"\n\nElectromagnetic Problem of Planar Array Antenna {parameters.n_x} X {parameters.n_y} Optimization for maximum Directivity and Side Lobe Level close to -20 dB \n")
        # Write details for each iteration
        for j in range(parameters.execute_iterations):
            file.write(
                f"Execution number {j}\nBest is {results.gbest_value[j]} at {results.gbest_bin[j]} found at iteration {results.iter_of_gbest[j]}_ directivity= {results.directivity[j]} sll={results.sll[j]}\n")

    # Compute additional statistics for the results
    best = min(results.gbest_value)
    worst = max(results.gbest_value)
    mean_value = statistics.mean(results.gbest_value)
    time_mean = statistics.mean(results.duration)

    # Check if there are multiple executions to calculate standard deviation
    if parameters.execute_iterations > 1:
        std_value = statistics.stdev(results.gbest_value) # Calculate standard deviation
        # Write the conclusion statistics
        file.write(f"\nmean={mean_value}\n")
        file.write(f"standard deviation={std_value}\n")
        file.write(f"best={best} at execution {results.gbest_value.index(best)}\n")
        file.write(f"worst={worst} at execution {results.gbest_value.index(worst)}\n")
        file.write(f"time mean={time_mean} sec\n")
        # Write fitness index and accuracy for non-electromagnetic problems
        if parameters.electromagnetic_problem_mode == 0:
            file.write(f"Parameter index ={ff_parameters_fitness_index}\n")
            file.write(f"Accuracy = {ff_parameters_accuracy} %\n")

    if parameters.parameters_selection_mode == 0 and parameters.electromagnetic_problem_mode == 1:
        # Store data for array-thinning problem optimization
        for i in range(parameters.execute_iterations):
            results.data_array.append(
                [results.gbest_bin[i], results.duration[i], results.gbest_value[i], results.directivity[i],
                 results.sll[i],
                 results.gbest_float_position[i]
                 ])

    elif parameters.parameters_selection_mode == 0 and parameters.electromagnetic_problem_mode == 0:
        # Store data for optimization of test functions
        results.data_array.append(
            [ff_name_array[parameters.ff_code], results.global_min_value, results.global_min, mean_value, std_value,
             best,
             worst, time_mean, ff_parameters_fitness_index, ff_parameters_accuracy])

    # Store data for parameter sets testing
    elif parameters.parameters_selection_mode == 1:
        row = [param_dictionary.get(key, 0) for key in param_dictionary.keys()]  # Retrieve values for all keys
        row.extend([round(ff_parameters_fitness_index, 4), round(ff_parameters_accuracy, 2)])  # Add fitness index and accuracy
        results.data_array.append(row)  # Add the row to the data array

        # Update optimum parameter set based on index
        if ff_parameters_fitness_index < results.optimum_parameter_set_index_based[-2]:
            results.optimum_parameter_set_index_based = row

        elif ff_parameters_fitness_index == results.optimum_parameter_set_index_based[-2]:
            if ff_parameters_accuracy > results.optimum_parameter_set_index_based[-1]:
                results.optimum_parameter_set_index_based = row

        # Update optimum parameter set based on accuracy
        if ff_parameters_accuracy > results.optimum_parameter_set_accuracy_based[-1]:
            results.optimum_parameter_set_accuracy_based = row

        elif ff_parameters_accuracy == results.optimum_parameter_set_accuracy_based[-1]:
            if ff_parameters_fitness_index < results.optimum_parameter_set_accuracy_based[-2]:
                results.optimum_parameter_set_accuracy_based = row

    return


# Function to edit text and data in the results file after all the iterations are done
def text_closure_editor(parameters: Parameters, results: Results, param_dictionary, file):
    # Write a newline character to separate sections in the file
    file.write("\n\n")

    if parameters.parameters_selection_mode == 1:
        # Initialize text dictionary for parameter selection mode
        text_dictionary = {
            "#": ""
        }
        text_dictionary.update(param_dictionary)
        text_dictionary['Index'] = ""
        text_dictionary['Accuracy'] = ""

        # Set delimiters for formatting
        delimiter1 = '\t&'
        delimiter2 = '\t&\t'

    elif parameters.parameters_selection_mode == 0 and parameters.electromagnetic_problem_mode == 0:
        # Initialize text dictionary for optimization of test functions
        text_dictionary = {
            "#": "",
            "Name": "",
            "Expected Min": "",
            "Mean Value": "",
            "Standard deviation": "",
            "Best": "",
            "Worst": "",
            "Time Mean": "",
            "Parameter Fit Index": "",
            "Accuracy": ""
        }
        # Set delimiters for formatting
        delimiter1 = ' & '
        delimiter2 = ' & '

    elif parameters.parameters_selection_mode == 0 and parameters.electromagnetic_problem_mode == 1:
        # Initialize text dictionary for electromagnetic problems
        text_dictionary = {
            "#": "",
            "Design": "",
            "Time": "",
            "FF Value": "",
            "Directivity": "",
            "SLL": ""
        }

        # Set delimiters for formatting
        delimiter1 = '\t&'
        delimiter2 = '\t&\t'
        # Sort the data array based on FF Value
        results.data_array.sort(key=lambda xx: (xx[2]), reverse=False)
        # Write column headers
        file.write(delimiter1.join(text_dictionary.keys()) + "\n")
        # Write data rows
        for counter, row in enumerate(results.data_array, start=1):
            # Exclude results.gbest_float_position from the row
            formatted_row = [str(counter)] + [f"{item:.2f}" if isinstance(item, float) else str(item) for index, item in
                                              enumerate(row) if index != 5]
            file.write(delimiter2.join(formatted_row) + "\\\ \hline \n")
        # End the function
        return

    # Write column headers
    file.write(delimiter1.join(text_dictionary.keys()) + "\n")
    # Write data rows
    for counter, row in enumerate(results.data_array, start=1):
        formatted_row = [str(counter)] + [f"{item:.2f}" if isinstance(item, float) else str(item) for item in row]
        file.write(delimiter2.join(formatted_row) + "\\\ \hline \n")

    file.write("\n\n")

    if parameters.parameters_selection_mode == 1:
        # Convert optimum parameter sets to strings
        index_str = ", ".join(map(str, results.optimum_parameter_set_index_based))
        accuracy_str = ", ".join(map(str, results.optimum_parameter_set_accuracy_based))

        # Write best parameter selections
        file.write(
            f"\nBest parameter selection based on index is {index_str}"
            f"\nBest parameter selection based on accuracy is {accuracy_str}"
        )
        file.write("\n\n")

    # Sort data array based on Accuracy and Parameter Fit Index
    sorted_data_array = sorted(results.data_array, key=lambda xx: (xx[-1], -xx[-2]), reverse=True)

    # Write sorted data array
    file.write(delimiter1.join(text_dictionary.keys()) + "\n")
    # Write data rows
    for counter, row in enumerate(sorted_data_array, start=1):
        formatted_row = [str(counter)] + [f"{item:.2f}" if isinstance(item, float) else str(item) for item in row]
        file.write(delimiter2.join(formatted_row) + "\\\ \hline \n")

    return


# Function to print terminal information to monitor progress of software
def terminal_infos(parameters: Parameters, param_dictionary, ff_name_array):
    if parameters.electromagnetic_problem_mode == 1:
        # In this case terminal infos must be provided inside the binary pso flow (through ray.log ), due to the use of ray framework
        return
    # Initialize an empty string to store terminal information
    str_info =""
    if parameters.parameters_selection_mode == 1:
    # Construct terminal information for parameter set currently checked
        str_info = "Parameters Check: "
        for key, value in param_dictionary.items():
            if isinstance(value, float):
                str_info += f"{key}\t{value:.2f}\t"
            else:
                str_info += f"{key}\t{value}\t"

    elif parameters.parameters_selection_mode == 0 and parameters.electromagnetic_problem_mode == 0:
        # Construct terminal information for test function currently in optimization process
        str_info = f"Optimization of function:{ff_name_array[parameters.ff_code]}"

    # Print the terminal information
    print(str_info, flush=True)
    return


def print_duration(duration_start):
    duration_seconds = timer() - duration_start
    # Calculate days, hours, minutes, and seconds
    days, remaining_seconds = divmod(duration_seconds, 86400)
    hours, remaining_seconds = divmod(remaining_seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)

    # Build the duration string
    duration_str = f"Total Duration: {int(days)} Days, {int(hours)} Hours, {int(minutes)} Minutes, {seconds:.1f} Seconds"

    # Print the duration string
    print(duration_str)
    return
