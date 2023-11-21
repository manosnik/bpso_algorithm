import numpy as np
from bitstring import BitArray
import random
import math
from scipy.stats import bernoulli


class Particle:

    def __init__(self, id, ff_code, bound_min, bound_max, number_of_decimals, omega_generator, c1_generator,
                 c2_generator,mutation_mode, v_ones_max_percentage):
        # bound_min ,bound_max coulb be determined inside Particle but if it's done in  Swarm it's faster

        self.id = id
        self.ff_code = ff_code
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.number_of_decimals = number_of_decimals

        self.omega_generator = omega_generator
        self.c1_generator = c1_generator
        self.c2_generator = c2_generator

        self.mutation_mode = mutation_mode
        self.v_ones_max_percentage = v_ones_max_percentage

        self.fitness_value = None
        self.fitness_value_p_best = np.inf

        self.bit_dim = 0
        self.compute_bit_dim()

        self.x = self.generate_random_bitstring(random_genarator=0.5)  # x is BitArray
        # print(f"first random pos {self.x.bin}")
        self.v = self.generate_random_bitstring(random_genarator=0.5)

        self.test_bitstring_in_bounds()

        self.p_best = self.x  # p_best is BitArray

    def __str__(self):
        return f"id:{self.id}\tx:{self.x.bin}\tfloat_pos:{self.float_position}\t fitness_value:{self.fitness_value}" \
               f"\tpbest:{self.p_best.bin}\t fitness_value_p_best:{self.fitness_value_p_best}"

    def compute_bit_dim(self):
        if self.ff_code == 1:  # Eggholder
            self.real_dim = 2
        elif self.ff_code == 2:  # Burkin 6
            self.real_dim = 2

        for i in range(self.real_dim):
            float_space = self.bound_max[i] - self.bound_min[i]
            self.bit_dim += math.floor(math.log2(float_space * math.pow(10, self.number_of_decimals))) + 1

        return

    def generate_random_bitstring(self, random_genarator):
        key1 = ""
        random_data = bernoulli.rvs(size=self.bit_dim, p=random_genarator)
        for i in range(self.bit_dim):
            temp = str(random_data[i])
            key1 += temp
        # print(key1)
        return BitArray(bin=key1)

        # tests if the position bitstring corresponds to a valid float position for the selected fitness function
        # if not it changes the position bitstring to the limits of valid space
        # In addition computes and saves the float position of the current position bitstring

    def test_bitstring_in_bounds(self):  # should be checked if self would be better to leave from some variables

        self.float_position = self.driver_bin_to_float(self.x)
        # print(self.float_position)
        flag = 0
        for i in range(self.real_dim):
            if self.float_position[i] > self.bound_max[i]:
                self.float_position[i] = self.bound_max[i]
                flag = 1
            elif self.float_position[i] < self.bound_min[i]:
                self.float_position[i] = self.bound_min[i]
                flag = 1
                print("alarm 1")
        if flag == 1:
            # print("out of bounds flag activated")
            self.x = self.driver_float_to_bin(self.float_position)
            # print(self.driver_bin_to_float(self.x))
        return

    def driver_bin_to_float(self, input_bitstring):
        output_float_array = np.zeros(len(self.bound_min))
        string_counter = 0
        for i in range(len(self.bound_min)):
            float_space = self.bound_max[i] - self.bound_min[i]
            number_of_bits_needed = math.floor(math.log2(float_space * math.pow(10, self.number_of_decimals))) + 1
            output = input_bitstring.bin[string_counter:string_counter + number_of_bits_needed]
            string_counter += number_of_bits_needed
            # print(output_float_array[i])
            # print(number_of_bits_needed)
            # print(output)
            output = int(output, base=2)

            # print("uint")
            # print(output_float_array[i])
            output = output / math.pow(10, self.number_of_decimals)
            output = output + self.bound_min[i]
            output_float_array[i] = output
            # print(output_float_array)
        return output_float_array

    def driver_float_to_bin(self, float_number_array):
        output_bitstring = BitArray()
        for i in range(len(float_number_array)):
            float_space = self.bound_max[i] - self.bound_min[i]
            float_number = float_number_array[i] - self.bound_min[i]
            number_of_bits_needed = math.floor(math.log2(float_space * math.pow(10, self.number_of_decimals))) + 1
            float_number = int(float_number * math.pow(10, self.number_of_decimals))
            output_bitstring.append(BitArray(uint=float_number, length=number_of_bits_needed))
        # print(f"dimensions of hybercube = {output_bitstring.len}")
        # print(output_bitstring)
        return output_bitstring

    def upgrade_vel(self, g_best):
        self.v = (self.generate_random_bitstring(self.omega_generator) & self.v) | (
                ((self.x ^ g_best) & self.generate_random_bitstring(self.c2_generator)) | (
                (self.x ^ self.p_best) & self.generate_random_bitstring(self.c1_generator)))
        if(self.mutation_mode==1):
            self.mutation()

        return

    def mutation(self):
        v_ones_max_number = math.ceil(self.v_ones_max_percentage * self.bit_dim)
        positions_of_ones = list(self.v.findall('0b1', bytealigned=0))
        number_of_subs = len(positions_of_ones) - v_ones_max_number
        if (number_of_subs > 0):
            positions_of_subs = random.sample(positions_of_ones, number_of_subs)
            self.v.invert(positions_of_subs)
            print("mutation")
        return

    def upgrade_position(self):
        self.x = self.x ^ self.v
        self.test_bitstring_in_bounds()
        return

    def upgrade_pbest(self):

        if (self.fitness_value < self.fitness_value_p_best):
            self.p_best = self.x
            self.fitness_value_p_best = self.fitness_value
        return

    def fitness_function(self):
        if self.ff_code == 1:  # Eggholder
            a = math.sqrt(math.fabs(self.float_position[1] + self.float_position[0] / 2 + 47))
            b = math.sqrt(math.fabs(self.float_position[0] - (self.float_position[1] + 47)))
            self.fitness_value = -(self.float_position[1] + 47) * math.sin(a) - self.float_position[0] * math.sin(b)

        elif self.ff_code == 2:  # Rastring

            d = len(self.bound_min)
            sum = 0
            for i in range(d):
                xi = self.float_position[i]
                sum = sum + (xi ** 2 - 10 * math.cos(2 * math.pi * xi))
            self.fitness_value = 10 * d + sum

        # term1 = 100 * math.sqrt(abs(self.float_position[1] - 0.01 * self.float_position[0] ** 2))
        # term2 = 0.01 * abs(self.float_position[0] + 10)
        # self.fitness_value = term1 + term2

        return

###TEST CODE
#
#
# bound_min = [-512] * 2
# bound_max = [512] * 2
#
# p = Particle(0, 1, bound_min, bound_max, 3)
#
# # print(p.float_position)
# for i in range(100):
#     p.fitness_function()
#     p.upgrade_pbest()
#     print(p)
#     p.upgrade_vel(p.generate_random_bitstring())
#     p.upgrade_position()
#     i+=1
