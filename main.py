from BinaryPso import BinaryPso
import numpy as np
import statistics
from timeit import default_timer as timer

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

# θέλουμε και διαστασεις για τισ συναρτησεις κυμαινόμενων διαστάσεων και ανα θελει αποτελέσματα σε csv or latex

#####################################

######  MANUAL  INPUT ##########

ff_code = 2
execute_iterations = 30
algorithm_iterations = 1000
swarm_size = 30
number_of_decimals = 5
omega_generator = 0.7
c1_generator = 0.6
c2_generator = 0.6
mutation_mode = 1
v_ones_max_percentage = 0.2

###############################

ff_name_array = ["--", "Eggholder", "Rastring"]

results = np.empty(execute_iterations)
results_time = np.empty(execute_iterations)

results_file = open("results.txt", "w")

for j in range(2):  # 2->Eggholder and Bukin6
    ff_code = j + 1  # begins from 1
    best = np.inf
    worst = -np.inf
    results_file.write(f"\n\n\t\t\t\t\t\t{ff_name_array[ff_code]}\n\n\n")
    for i in range(execute_iterations):
        results_file.write(f"Execute number ={i}\n")
        start = timer()
        algorithm = BinaryPso(size=swarm_size, ff_code=ff_code, number_of_decimals=number_of_decimals,
                              omega_generator=omega_generator, c1_generator=c1_generator, c2_generator=c2_generator,
                              mutation_mode=mutation_mode, v_ones_max_percentage=v_ones_max_percentage,
                              algorithm_iterations=algorithm_iterations, file=results_file)
        results_time[i] = timer() - start
        results[i] = algorithm.swarm.gbest_value
        if algorithm.swarm.gbest_value < best:
            best = algorithm.swarm.gbest_value
            golden_iteration = i
        if algorithm.swarm.gbest_value > worst:
            worst = algorithm.swarm.gbest_value
            worst_iteration = i

        # results_file.write(f"{results[i]}\n")

    results_file.write(f"\n\n\t\t\t\t\tmean={statistics.mean(results)}\n")
    results_file.write(f"\n\n\t\t\t\t\tstandard deviation={statistics.stdev(results)}\n")
    results_file.write(f"\n\n\t\t\t\t\tbest={best} at iteration {golden_iteration}\n")
    results_file.write(f"\n\n\t\t\t\t\tworst={worst} at iteration {worst_iteration}\n")
    results_file.write(f"\n\n\t\t\t\t\ttime mean={statistics.mean(results_time)} sec\n")

    # plus time

results_file.close()
