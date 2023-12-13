import math
from BinaryPso import BinaryPso
import numpy as np
import statistics
from timeit import default_timer as timer
import multiprocessing
from Results import Results
from Parameters import Parameters


def run_alg(parameters):
    start = timer()
    algorithm = BinaryPso(size=parameters.size, ff_code=parameters.ff_code,
                          number_of_decimals=parameters.number_of_decimals,
                          omega_generator=parameters.omega_generator, c1_generator=parameters.c1_generator,
                          c2_generator=parameters.c2_generator,
                          mutation_mode=parameters.mutation_mode,
                          v_ones_max_percentage=parameters.v_ones_max_percentage,
                          algorithm_iterations=parameters.algorithm_iterations,
                          dim_of_multidim_fun=parameters.dim_of_multidim_fun)
    fitness_index_contribution = math.sqrt(
        (algorithm.swarm.global_min[0] - algorithm.swarm.g_best_float_position[0]) ** 2 + (
                algorithm.swarm.global_min[1] - algorithm.swarm.g_best_float_position[
            1]) ** 2)
    results = Results(results_time=timer() - start, results=algorithm.swarm.gbest_value,
                      results_g_best_bin=algorithm.swarm.g_best.bin,
                      results_g_best_float_position=algorithm.swarm.g_best_float_position,
                      results_iter_of_g_best=algorithm.iteration_of_gbest,
                      results_parameter_fitness_index=fitness_index_contribution)
    return results


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


ff_code = 6
execute_iterations = 10
algorithm_iterations = 1000
swarm_size = 50
number_of_decimals = 4
omega_generator = 0.5
c1_generator = 0.5
c2_generator = 0.5  # omega c1 c2 [0,1]
mutation_mode = 1
v_ones_max_percentage = 0.4
dim_of_multidim_fun = 5
parallel_mode = 1
parameters_selection_mode = 1
###############################


if __name__ == '__main__':
    total_time_start = timer()

    # TODO complete ff list
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
                     "Michalewicz Dim: " + str(dim_of_multidim_fun), "Colville"]

    if parameters_selection_mode == 0:
        results_file = open("results.txt", "w")
        results = []
        results_time = []
        results_g_best_bin = []
        results_g_best_float_position = []
        results_iter_of_g_best = []

        for ff_code in range(1, 3):

            if parallel_mode == 0:
                for i in range(execute_iterations):
                    start = timer()
                    algorithm = BinaryPso(size=swarm_size, ff_code=ff_code, number_of_decimals=number_of_decimals,
                                          omega_generator=omega_generator, c1_generator=c1_generator,
                                          c2_generator=c2_generator,
                                          mutation_mode=mutation_mode, v_ones_max_percentage=v_ones_max_percentage,
                                          algorithm_iterations=algorithm_iterations,
                                          dim_of_multidim_fun=dim_of_multidim_fun)
                    results_time.append(timer() - start)
                    results.append(algorithm.swarm.gbest_value)
                    results_g_best_bin.append(algorithm.swarm.g_best.bin)
                    results_g_best_float_position.append(algorithm.swarm.g_best_float_position)
                    results_iter_of_g_best.append(algorithm.iteration_of_gbest)

            elif parallel_mode == 1:
                pool = multiprocessing.Pool(processes=4)

                input = []
                input_parameters = Parameters(size=swarm_size, ff_code=ff_code, number_of_decimals=number_of_decimals,
                                              omega_generator=omega_generator, c1_generator=c1_generator,
                                              c2_generator=c2_generator,
                                              mutation_mode=mutation_mode, v_ones_max_percentage=v_ones_max_percentage,
                                              algorithm_iterations=algorithm_iterations,
                                              dim_of_multidim_fun=dim_of_multidim_fun)
                for j in range(execute_iterations):
                    input.append(input_parameters)

                output = pool.map(run_alg, input)

                for j in range(execute_iterations):
                    results.append(output[j].results)
                    results_time.append(output[j].results_time)
                    results_g_best_bin.append(output[j].results_g_best_bin)
                    results_g_best_float_position.append(output[j].results_g_best_float_position)
                    results_iter_of_g_best.append(output[j].results_iter_of_g_best)

            results_file.write(f"\n\n{ff_name_array[ff_code]}\n\n")
            for j in range(execute_iterations):
                results_file.write(
                    f"Execution number {j}\nBest is {results[j]} at {results_g_best_bin[j]} or {results_g_best_float_position[j]} found at iteration {results_iter_of_g_best[j]}\n")

            if execute_iterations > 1:
                results_file.write(f"\nmean={statistics.mean(results)}\n")
                results_file.write(f"standard deviation={statistics.stdev(results)}\n")
                results_file.write(f"best={min(results)} at iteration {results.index(min(results))}\n")
                results_file.write(f"worst={max(results)} at iteration {results.index(max(results))}\n")
                results_file.write(f"time mean={statistics.mean(results_time)} sec\n")

            results.clear()
            results_time.clear()
            results_g_best_bin.clear()
            results_g_best_float_position.clear()
            results_iter_of_g_best.clear()

    # TODO parallelize parameter selection mode and retuse it
    if parameters_selection_mode == 1:
        results_file = open("results.txt", "w")
        results = []
        results_time = []
        results_g_best_bin = []
        results_g_best_float_position = []
        results_iter_of_g_best = []
        results_parameter_fitness_index = []

        optimum_parameter_set = np.empty(5)
        optimum_parameter_set[0] = np.inf
        parameter_set_array = []
        for r in range(4):
            swarm_size = 30 + 10 * r
            for e in range(1, 2):  # range(1, 5)->1,2,3,4
                v_ones_max_percentage = 0.05 * e
                for k in range(1, 2):
                    omega_generator = 0.05 * k
                    for h in range(1, 2):
                        c1_generator = 0.1 * h
                        c2_generator = c1_generator

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

                                fitness_index_contribution = math.sqrt(
                                    (algorithm.swarm.global_min[0] - algorithm.swarm.g_best_float_position[0]) ** 2 + (
                                            algorithm.swarm.global_min[1] - algorithm.swarm.g_best_float_position[
                                        1]) ** 2)

                                results_time.append(timer() - start)
                                results.append(algorithm.swarm.gbest_value)
                                results_g_best_bin.append(algorithm.swarm.g_best.bin)
                                results_g_best_float_position.append(algorithm.swarm.g_best_float_position)
                                results_iter_of_g_best.append(algorithm.iteration_of_gbest)
                                results_parameter_fitness_index.append(fitness_index_contribution)

                        elif parallel_mode == 1:
                            pool = multiprocessing.Pool(processes=4)

                            input = []
                            input_parameters = Parameters(size=swarm_size, ff_code=ff_code,
                                                          number_of_decimals=number_of_decimals,
                                                          omega_generator=omega_generator, c1_generator=c1_generator,
                                                          c2_generator=c2_generator,
                                                          mutation_mode=mutation_mode,
                                                          v_ones_max_percentage=v_ones_max_percentage,
                                                          algorithm_iterations=algorithm_iterations,
                                                          dim_of_multidim_fun=dim_of_multidim_fun)
                            for j in range(execute_iterations):
                                input.append(input_parameters)

                            output = pool.map(run_alg, input)

                            for j in range(execute_iterations):
                                results.append(output[j].results)
                                results_time.append(output[j].results_time)
                                results_g_best_bin.append(output[j].results_g_best_bin)
                                results_g_best_float_position.append(output[j].results_g_best_float_position)
                                results_iter_of_g_best.append(output[j].results_iter_of_g_best)
                                results_parameter_fitness_index.append(output[j].results_parameter_fitness_index)

                        results_file.write(f"\n\n{ff_name_array[ff_code]}\n\n")
                        for j in range(execute_iterations):
                            results_file.write(
                                f"Execution number {j}\nBest is {results[j]} at {results_g_best_bin[j]} or {results_g_best_float_position[j]} found at iteration {results_iter_of_g_best[j]}\n")

                        ff_parameters_fitness_index=sum(results_parameter_fitness_index)
                        if execute_iterations > 1:
                            results_file.write(f"\nmean={statistics.mean(results)}\n")
                            results_file.write(f"standard deviation={statistics.stdev(results)}\n")
                            results_file.write(f"best={min(results)} at iteration {results.index(min(results))}\n")
                            results_file.write(f"worst={max(results)} at iteration {results.index(max(results))}\n")
                            results_file.write(f"time mean={statistics.mean(results_time)} sec\n")
                            results_file.write(f"Parameter index ={ff_parameters_fitness_index}\n")

                        parameter_set_array.append(
                            [ff_parameters_fitness_index, swarm_size, v_ones_max_percentage, omega_generator, c1_generator])

                        if ff_parameters_fitness_index < optimum_parameter_set[0]:
                            optimum_parameter_set[0] = ff_parameters_fitness_index
                            optimum_parameter_set[1] = swarm_size
                            optimum_parameter_set[2] = v_ones_max_percentage
                            optimum_parameter_set[3] = omega_generator
                            optimum_parameter_set[4] = c1_generator

                        print(
                            f"swarm size ={swarm_size}, v_ones max = {round(v_ones_max_percentage, 2)} , omega={round(omega_generator, 2)},c={round(c2_generator, 2)} execution  completed\n\n")

                        results.clear()
                        results_time.clear()
                        results_g_best_bin.clear()
                        results_g_best_float_position.clear()
                        results_iter_of_g_best.clear()
                        results_parameter_fitness_index.clear()


        print(parameter_set_array)



        results_file.write(
            "\n\n  & \en Swarm size \gr & \en Mutation \gr & \en Omega generator\gr  &$ c_{1} , c_{2}$ \en generator \gr & \en Parameter Fit Index\gr \\\ \hline \hline \n")
        counter = 1
        for row in parameter_set_array:
            results_file.write(
                f"{counter} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {round(row[0], 3)}\\\ \hline \n")
            counter += 1

        results_file.write(f"\nBest parameter selection is {optimum_parameter_set}")


    results_file.close()

    total_time = timer() - total_time_start
    print(f"{total_time} sec")
