import math
from BinaryPso import BinaryPso
import numpy as np
import statistics
from timeit import default_timer as timer
import multiprocessing
from Results import Results
from Parameters import Parameters
from scipy.spatial import distance

import os

# Set RAY_DEDUP_LOGS environment variable
os.environ["RAY_DEDUP_LOGS"] = "0"

import ray
from ray.util import inspect_serializability

####### TERMINAL INPUT  ############

# print("Enter Fitness Function Code (ff_code)")
# ff_code = int(input())
#
# print("Enter Execute Iterations")  #How many times to execute the algorithm for every fitness function
# execute_iterations = int(input())
#
# print("Enter Algorithm Iterations")
# algorithm_iterations = int(input())
#
# print("Enter Swarm Size")
# swarm_size = int(input())
#
# print("Enter Number of Decimals")
# number_of_decimals = int(input())
#
# print("Enter Generator for Inertia (Omega)")
# omega_generator = int(input())
#
# print("Enter Generator for C1")
# c1_generator = int(input())
#
# print("Enter Generator for C2")
# c2_generator = int(input())
#
# print("Mutation Mode: on->1 off->0")
# mutation_mode = int(input())
#
# if(mutation_mode==1):
#   print("Enter Max Percentage of ones for velocity")
#   v_ones_max_percentage = int(input())
#
# print("Enter dimensions for multi-dimensional functions")
# dim_of_multidim_fun = int(input())

####################################

######  MANUAL  INPUT ##########

execute_iterations = 10
algorithm_iterations = 1000
number_of_decimals = 4
dim_of_multidim_fun = 5

mutation_mode = 1
parallel_mode = 1
number_of_processes = 8
parameters_selection_mode = 0  # if its 0 ff_code matters but the rest not if its 1 ff_code doesn't matter but the rest do
accuracy_threshold_per_dim = 0.5

ff_code = 7
swarm_size = 50
v_ones_max_percentage = 0.4
omega_generator = 0.5
c1_generator = 0.5
c2_generator = 0.5  # omega c1 c2 [0,1]


###############################

@ray.remote
def run_alg(parameters):
    start = timer()
    # print("Entered remote func")
    algorithm = BinaryPso(size=parameters.size, ff_code=parameters.ff_code,
                          number_of_decimals=parameters.number_of_decimals,
                          omega_generator=parameters.omega_generator, c1_generator=parameters.c1_generator,
                          c2_generator=parameters.c2_generator,
                          mutation_mode=parameters.mutation_mode,
                          v_ones_max_percentage=parameters.v_ones_max_percentage,
                          algorithm_iterations=parameters.algorithm_iterations,
                          dim_of_multidim_fun=parameters.dim_of_multidim_fun)

    fitness_index_contribution = None
    results_accuracy_hit = None

    if algorithm.swarm.global_min is not None:
        fitness_index_contribution = min(
            [distance.euclidean(point, algorithm.swarm.g_best_float_position) for point in algorithm.swarm.global_min])
        if fitness_index_contribution < accuracy_threshold_per_dim * len(algorithm.swarm.bound_min):
            results_accuracy_hit = 1
        else:
            results_accuracy_hit = 0

    results = Results(results_time=timer() - start, results=algorithm.swarm.gbest_value,
                      results_g_best_bin=algorithm.swarm.g_best.bin,
                      results_g_best_float_position=algorithm.swarm.g_best_float_position,
                      results_iter_of_g_best=algorithm.iteration_of_gbest,
                      results_parameter_fitness_index=fitness_index_contribution,
                      results_accuracy_hit=results_accuracy_hit, results_global_min=algorithm.swarm.global_min,
                      results_global_min_value=algorithm.swarm.global_min_value)
    # print("Finished function")

    return results


if __name__ == '__main__':
    print(ray.__version__)
    total_time_start = timer()

    # TODO make the ff list alphabetical

    ff_name_array = ["--", "Eggholder", "Rastring", "Ackley (a=20 , b=0.2 , c=2*pi) Dim: " + str(dim_of_multidim_fun),
                     "Rosenbrock Dim: " + str(dim_of_multidim_fun), "Dropwave", "Bukin N.6", "De Jong N.5",
                     "Schwefel Dim: " + str(dim_of_multidim_fun), "Cross-in-Tray", "Holder Table"
        , "Bohachevsky", "Griewank Dim: " + str(dim_of_multidim_fun), "Easom",
                     "Dixon-Price Dim: " + str(dim_of_multidim_fun), "Six-Hump Camel", "Gramacy & Lee Function"
        , "Shubert", "Langermann", "Schaffer N.2", "Trid Dim: " + str(dim_of_multidim_fun), "Sphere",
                     "Sum Squares function Dim: " + str(dim_of_multidim_fun),
                     "Sum of Different Powers function Dim: " + str(dim_of_multidim_fun)
        , "Rotated Hyper-Ellipsoid Dim: " + str(dim_of_multidim_fun), "Booth", "Matyas", "McCormick",
                     "Levy Dim: " + str(dim_of_multidim_fun), "Levy N.13", "Schaffer N.4", "Power Sum Function"
        , "Zakharov Dim: " + str(dim_of_multidim_fun), "Three-Hump Camel", "Beale", "Branin",
                     "Michalewicz Dim: " + str(dim_of_multidim_fun), "Colville", "Shekel", "Styblinski-Tang", "Powell",
                     "Goldstein-Price", "Forrester", "Perm d,b", "Perm 0,d,b",
                     "Hartmann 3-D", "Hartmann 4-D", "Hartmann 6-D"]

    # s=sorted(ff_name_array)
    # for i in range(len(s)):
    #     print(f"{s[i]}\n" )

    if parameters_selection_mode == 0:
        results_file = open("results.txt", "w")
        results = []
        results_time = []
        results_g_best_bin = []
        results_g_best_float_position = []
        results_iter_of_g_best = []
        results_parameter_fitness_index = []
        results_accuracy_hit_array = []
        results_global_min = None
        results_global_min_value = None

        final_results_array = []

        if parallel_mode == 1:
            ray.init(num_cpus=number_of_processes)
            print(ray.available_resources(),flush=True)

        for ff_code in range(7, 10):
            print(f"Optimization of function:{ff_name_array[ff_code]}", flush=True)
            if parallel_mode == 0:
                for i in range(execute_iterations):
                    start = timer()
                    algorithm = BinaryPso(size=swarm_size, ff_code=ff_code, number_of_decimals=number_of_decimals,
                                          omega_generator=omega_generator, c1_generator=c1_generator,
                                          c2_generator=c2_generator,
                                          mutation_mode=mutation_mode, v_ones_max_percentage=v_ones_max_percentage,
                                          algorithm_iterations=algorithm_iterations,
                                          dim_of_multidim_fun=dim_of_multidim_fun)

                    fitness_index_contribution = None
                    results_accuracy_hit = None

                    if algorithm.swarm.global_min is not None:
                        fitness_index_contribution = min(
                            [distance.euclidean(point, algorithm.swarm.g_best_float_position) for point in
                             algorithm.swarm.global_min])
                        if fitness_index_contribution < accuracy_threshold_per_dim * len(algorithm.swarm.bound_min):
                            results_accuracy_hit = 1
                        else:
                            results_accuracy_hit = 0

                    results_time.append(timer() - start)
                    results.append(algorithm.swarm.gbest_value)
                    results_g_best_bin.append(algorithm.swarm.g_best.bin)
                    results_g_best_float_position.append(algorithm.swarm.g_best_float_position)
                    results_iter_of_g_best.append(algorithm.iteration_of_gbest)
                    results_parameter_fitness_index.append(fitness_index_contribution)
                    results_accuracy_hit_array.append(results_accuracy_hit)
                    if i == 0:
                        results_global_min = algorithm.swarm.global_min
                        results_global_min_value = algorithm.swarm.global_min_value



            elif parallel_mode == 1:

                input_parameters = Parameters(size=swarm_size, ff_code=ff_code, number_of_decimals=number_of_decimals,
                                              omega_generator=omega_generator, c1_generator=c1_generator,
                                              c2_generator=c2_generator,
                                              mutation_mode=mutation_mode, v_ones_max_percentage=v_ones_max_percentage,
                                              algorithm_iterations=algorithm_iterations,
                                              dim_of_multidim_fun=dim_of_multidim_fun)

                input_ray = ray.put(input_parameters)
                results_ray = ray.get([run_alg.remote(input_ray) for _ in range(execute_iterations)])

                for j in range(execute_iterations):
                    results.append(results_ray[j].results)
                    results_time.append(results_ray[j].results_time)
                    results_g_best_bin.append(results_ray[j].results_g_best_bin)
                    results_g_best_float_position.append(results_ray[j].results_g_best_float_position)
                    results_iter_of_g_best.append(results_ray[j].results_iter_of_g_best)
                    results_parameter_fitness_index.append(results_ray[j].results_parameter_fitness_index)
                    results_accuracy_hit_array.append(results_ray[j].results_accuracy_hit)
                results_global_min = results_ray[0].results_global_min
                results_global_min_value = results_ray[0].results_global_min_value

            #print(results_accuracy_hit_array)
            # Results editing and presentation

            if results_global_min is None:
                # results_global_min=results_g_best_float_position[results.index(min(results))]
                if results_global_min_value is None:
                    results_global_min_value = round(min(results), 4)
                for j in range(execute_iterations):
                    results_parameter_fitness_index[j] = results[j] - results_global_min_value
                    if results_parameter_fitness_index[j] < accuracy_threshold_per_dim:
                        results_accuracy_hit_array[j] = 1
                    else:
                        results_accuracy_hit_array[j] = 0

            results_file.write(
                f"\n\n{ff_name_array[ff_code]} Global Minimum = {results_global_min_value} at {results_global_min}\n\n")
            for j in range(execute_iterations):
                results_file.write(
                    f"Execution number {j}\nBest is {results[j]} at {results_g_best_bin[j]} or {results_g_best_float_position[j]} found at iteration {results_iter_of_g_best[j]}_ index= {results_parameter_fitness_index[j]} hit={results_accuracy_hit_array[j]}\n")

            ff_parameters_fitness_index = sum(results_parameter_fitness_index)
            ff_parameters_accuracy = (sum(results_accuracy_hit_array) / execute_iterations) * 100
            best = min(results)
            worst = max(results)
            mean_value = statistics.mean(results)
            time_mean = statistics.mean(results_time)
            if execute_iterations > 1:
                std_value = statistics.stdev(results)
                results_file.write(f"\nmean={mean_value}\n")
                results_file.write(f"standard deviation={std_value}\n")
                results_file.write(f"best={best} at execution {results.index(min(results))}\n")
                results_file.write(f"worst={worst} at execution {results.index(max(results))}\n")
                results_file.write(f"time mean={time_mean} sec\n")
                results_file.write(f"Parameter index ={ff_parameters_fitness_index}\n")
                results_file.write(f"Accuracy = {ff_parameters_accuracy} %\n")

            final_results_array.append(
                [ff_name_array[ff_code], results_global_min_value, results_global_min, mean_value, std_value, best,
                 worst, time_mean, ff_parameters_fitness_index, ff_parameters_accuracy])

            results.clear()
            results_time.clear()
            results_g_best_bin.clear()
            results_g_best_float_position.clear()
            results_iter_of_g_best.clear()
            results_parameter_fitness_index.clear()
            results_accuracy_hit_array.clear()
        results_file.write(
            "\n\n # & \en Name \gr & \en Expected Min \gr & \en Mean Value \gr  & \en Standard deviation  \gr & \en Best \gr & \en Worst \gr & \en Time Mean \gr & \en Parameter Fit Index\gr & \en Accuracy \gr \\\ \hline \hline \n")
        counter = 1
        for row in final_results_array:
            results_file.write(
                f"{counter} & {row[0]} & {row[1]} at {row[2]} & {round(row[3], 4)} & {round(row[4], 4)} & {round(row[5], 4)} & {round(row[6], 4)} & {round(row[7], 4)} & {round(row[8], 4)} & {row[9]}\\\ \hline \n")
            counter += 1
    # TODO parallelize parameter selection mode and retuse it
    if parameters_selection_mode == 1:
        results_file = open("results.txt", "w")
        results = []
        results_time = []
        results_g_best_bin = []
        results_g_best_float_position = []
        results_iter_of_g_best = []
        results_parameter_fitness_index = []
        results_accuracy_hit_array = []
        results_global_min = None

        optimum_parameter_set = np.empty(6)
        optimum_parameter_set[0] = np.inf

        optimum_parameter_set_accuracy_based = np.empty(6)
        optimum_parameter_set_accuracy_based[5] = -1

        parameter_set_array = []

        if parallel_mode == 1:
            ray.init(num_cpus=number_of_processes)
            print(ray.available_resources(),flush=True)



        for swarm_size in range(30, 40, 10):
            for e in range(1, 2):  # range(1, 5)->1,2,3,4
                v_ones_max_percentage = 0.1 * e
                for k in range(1, 2):
                    omega_generator = 0.1 * k
                    for h in range(1, 5):
                        c1_generator = 0.2 * h
                        c2_generator = c1_generator
                        print(
                            f"Parameters Check : swarm size ={swarm_size}  , v_ones max = {round(v_ones_max_percentage, 2)}  ,  omega={round(omega_generator, 2)}  ,  c={round(c2_generator, 2)}",
                            flush=True)

                        if parallel_mode == 0:
                            for i in range(execute_iterations):
                                start = timer()
                                algorithm = BinaryPso(size=swarm_size, ff_code=ff_code,
                                                      number_of_decimals=number_of_decimals,
                                                      omega_generator=omega_generator, c1_generator=c1_generator,
                                                      c2_generator=c2_generator,
                                                      mutation_mode=mutation_mode,
                                                      v_ones_max_percentage=v_ones_max_percentage,
                                                      algorithm_iterations=algorithm_iterations,
                                                      dim_of_multidim_fun=dim_of_multidim_fun)

                                fitness_index_contribution = None
                                results_accuracy_hit = None

                                if algorithm.swarm.global_min is not None:
                                    fitness_index_contribution = min(
                                        [distance.euclidean(point, algorithm.swarm.g_best_float_position) for point in
                                         algorithm.swarm.global_min])
                                    if fitness_index_contribution < accuracy_threshold_per_dim * len(
                                            algorithm.swarm.bound_min):
                                        results_accuracy_hit = 1
                                    else:
                                        results_accuracy_hit = 0

                                results_time.append(timer() - start)
                                results.append(algorithm.swarm.gbest_value)
                                results_g_best_bin.append(algorithm.swarm.g_best.bin)
                                results_g_best_float_position.append(algorithm.swarm.g_best_float_position)
                                results_iter_of_g_best.append(algorithm.iteration_of_gbest)
                                results_parameter_fitness_index.append(fitness_index_contribution)
                                results_accuracy_hit_array.append(results_accuracy_hit)
                                if i == 0:
                                    results_global_min = algorithm.swarm.global_min
                                    results_global_min_value = algorithm.swarm.global_min_value

                        elif parallel_mode == 1:
                            # pool = multiprocessing.Pool(processes=number_of_processes)

                            # input = []
                            input_parameters = Parameters(size=swarm_size, ff_code=ff_code,
                                                          number_of_decimals=number_of_decimals,
                                                          omega_generator=omega_generator, c1_generator=c1_generator,
                                                          c2_generator=c2_generator,
                                                          mutation_mode=mutation_mode,
                                                          v_ones_max_percentage=v_ones_max_percentage,
                                                          algorithm_iterations=algorithm_iterations,
                                                          dim_of_multidim_fun=dim_of_multidim_fun)

                            # inspect_serializability(input_parameters, name="input_parameters")

                            # for j in range(execute_iterations):
                            #   input.append(input_parameters)

                            input_ray = ray.put(input_parameters)
                            results_ray = ray.get(
                                [run_alg.remote(input_ray) for _ in range(execute_iterations)])

                            # print(results_ray)

                            # output = pool.map(run_alg, input)

                            for j in range(execute_iterations):
                                results.append(results_ray[j].results)
                                results_time.append(results_ray[j].results_time)
                                results_g_best_bin.append(results_ray[j].results_g_best_bin)
                                results_g_best_float_position.append(results_ray[j].results_g_best_float_position)
                                results_iter_of_g_best.append(results_ray[j].results_iter_of_g_best)
                                results_parameter_fitness_index.append(results_ray[j].results_parameter_fitness_index)
                                results_accuracy_hit_array.append(results_ray[j].results_accuracy_hit)
                            results_global_min = results_ray[0].results_global_min
                            results_global_min_value = results_ray[0].results_global_min_value

                        # Results editing and presentation

                        # print(results)
                        if results_global_min is None:
                            # results_global_min=results_g_best_float_position[results.index(min(results))]
                            results_global_min_value = round(min(results), 4)
                            for j in range(execute_iterations):
                                results_parameter_fitness_index[j] = results[j] - results_global_min_value
                                if results_parameter_fitness_index[j] < accuracy_threshold_per_dim:
                                    results_accuracy_hit_array[j] = 1
                                else:
                                    results_accuracy_hit_array[j] = 0

                        results_file.write(
                            f"\n\n{ff_name_array[ff_code]} (global min ={results_global_min_value} at {results_global_min})swarm size ={swarm_size}, v_ones max = {round(v_ones_max_percentage, 2)} , omega={round(omega_generator, 2)},c={round(c2_generator, 2)} execution  completed\n\n")
                        for j in range(execute_iterations):
                            results_file.write(
                                f"Execution number {j}\nBest is {results[j]} at {results_g_best_bin[j]} or {results_g_best_float_position[j]} found at iteration {results_iter_of_g_best[j]} _ index= {results_parameter_fitness_index[j]} hit={results_accuracy_hit_array[j]}\n")

                        ff_parameters_fitness_index = sum(results_parameter_fitness_index)
                        ff_parameters_accuracy = (sum(results_accuracy_hit_array) / execute_iterations) * 100
                        if execute_iterations > 1:
                            results_file.write(f"\nmean={statistics.mean(results)}\n")
                            results_file.write(f"standard deviation={statistics.stdev(results)}\n")
                            results_file.write(f"best={min(results)} at execution {results.index(min(results))}\n")
                            results_file.write(f"worst={max(results)} at execution {results.index(max(results))}\n")
                            results_file.write(f"time mean={statistics.mean(results_time)} sec\n")
                            results_file.write(f"Parameter index ={ff_parameters_fitness_index}\n")
                            results_file.write(f"Accuracy = {ff_parameters_accuracy} %\n")

                        parameter_set_array.append(
                            [round(ff_parameters_fitness_index, 4), swarm_size, round(v_ones_max_percentage, 2),
                             round(omega_generator, 2),
                             round(c1_generator, 2), round(ff_parameters_accuracy, 2)])

                        if ff_parameters_fitness_index < optimum_parameter_set[0]:
                            optimum_parameter_set[0] = round(ff_parameters_fitness_index, 4)
                            optimum_parameter_set[1] = swarm_size
                            optimum_parameter_set[2] = round(v_ones_max_percentage, 2)
                            optimum_parameter_set[3] = round(omega_generator, 2)
                            optimum_parameter_set[4] = round(c1_generator, 2)
                            optimum_parameter_set[5] = round(ff_parameters_accuracy, 2)

                        if ff_parameters_accuracy > optimum_parameter_set_accuracy_based[5]:
                            optimum_parameter_set_accuracy_based[0] = round(ff_parameters_fitness_index, 4)
                            optimum_parameter_set_accuracy_based[1] = swarm_size
                            optimum_parameter_set_accuracy_based[2] = round(v_ones_max_percentage, 2)
                            optimum_parameter_set_accuracy_based[3] = round(omega_generator, 2)
                            optimum_parameter_set_accuracy_based[4] = round(c1_generator, 2)
                            optimum_parameter_set_accuracy_based[5] = round(ff_parameters_accuracy, 2)

                        results.clear()
                        results_time.clear()
                        results_g_best_bin.clear()
                        results_g_best_float_position.clear()
                        results_iter_of_g_best.clear()
                        results_parameter_fitness_index.clear()
                        results_accuracy_hit_array.clear()

        # print(parameter_set_array)

        results_file.write(
            "\n\n  & \en Swarm size \gr & \en Mutation \gr & \en Omega generator\gr  &$ c_{1} , c_{2}$ \en generator \gr & \en Parameter Fit Index\gr & \en Accuracy \gr \\\ \hline \hline \n")
        counter = 1
        for row in parameter_set_array:
            results_file.write(
                f"{counter} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {round(row[0], 3)} & {row[5]}\\\ \hline \n")
            counter += 1

        results_file.write(
            f"\nBest parameter selection is {optimum_parameter_set[1]},  {optimum_parameter_set[2]},  {optimum_parameter_set[3]},  {optimum_parameter_set[4]},  {optimum_parameter_set[0]},  {optimum_parameter_set[5]}")
        results_file.write(
            f"\nBest parameter selection based on accuracy is {optimum_parameter_set_accuracy_based[1]},  {optimum_parameter_set_accuracy_based[2]},  {optimum_parameter_set_accuracy_based[3]},  {optimum_parameter_set_accuracy_based[4]},  {optimum_parameter_set_accuracy_based[0]},  {optimum_parameter_set_accuracy_based[5]}")

    if parallel_mode == 1:
        ray.shutdown()
    results_file.close()

    total_time = timer() - total_time_start
    print(f"{total_time} sec")
