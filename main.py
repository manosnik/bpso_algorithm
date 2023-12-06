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
# print("Enter dimensions for multi-dimensional functions")
# dim_of_multidim_fun = int(input())

#####################################

######  MANUAL  INPUT ##########

ff_code = 2
execute_iterations = 10
algorithm_iterations = 100
swarm_size = 500
number_of_decimals = 6
omega_generator = 0.6
c1_generator = 0.4
c2_generator = 0.4        # omega c1 c2 [0,1]
mutation_mode = 1
v_ones_max_percentage = 0.2
dim_of_multidim_fun = 3

###############################

ff_name_array = ["--", "Eggholder", "Rastring", "Ackley (a=20 , b=0.2 , c=2*pi) Dim: " + str(dim_of_multidim_fun),
                 "Rosenbrock Dim: " + str(dim_of_multidim_fun), "Dropwave", "Bukin N.6", "De Jong N.5","Schwefel Dim: " + str(dim_of_multidim_fun),"Cross-in-Tray" ,"Holder Table"
                 ,"Bohachevsky" , "Griewank Dim: " + str(dim_of_multidim_fun), "Easom" ,  "Dixon-Price Dim: " + str(dim_of_multidim_fun),"Six-Hump Camel","Gramacy & Lee Function"
                 ,"Shubert","Langermann","Schaffer N.2" , "Trid Dim: " + str(dim_of_multidim_fun), "Sphere" , "Sum Squares function Dim: " + str(dim_of_multidim_fun) ,"Sum of Different Powers function Dim: " + str(dim_of_multidim_fun)
                 ,"Rotated Hyper-Ellipsoid Dim: " + str(dim_of_multidim_fun)]

results = np.empty(execute_iterations)
results_time = np.empty(execute_iterations)

results_file = open("results.txt", "w")

total_time_start = timer()

# for j in range(len(ff_name_array) - 1):
for j in range(23, 24):  # range(1, 5)->1,2,3,4
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
                              algorithm_iterations=algorithm_iterations, file=results_file,
                              dim_of_multidim_fun=dim_of_multidim_fun)
        results_time[i] = timer() - start
        results[i] = algorithm.swarm.gbest_value
        if algorithm.swarm.gbest_value < best:
            best = algorithm.swarm.gbest_value
            golden_iteration = i
        if algorithm.swarm.gbest_value > worst:
            worst = algorithm.swarm.gbest_value
            worst_iteration = i
        print(f"{ff_name_array[ff_code]} execution {i} completed")
        # results_file.write(f"{results[i]}\n")

    if execute_iterations > 1:
        results_file.write(f"\n\n\t\t\t\t\tmean={statistics.mean(results)}\n")
        results_file.write(f"\n\n\t\t\t\t\tstandard deviation={statistics.stdev(results)}\n")
        results_file.write(f"\n\n\t\t\t\t\tbest={best} at iteration {golden_iteration}\n")
        results_file.write(f"\n\n\t\t\t\t\tworst={worst} at iteration {worst_iteration}\n")
        results_file.write(f"\n\n\t\t\t\t\ttime mean={statistics.mean(results_time)} sec\n")

results_file.close()

total_time = timer() - total_time_start
print(f"{total_time} sec")
