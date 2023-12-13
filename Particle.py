import numpy as np
from bitstring import BitArray
import random
import math
from scipy.stats import bernoulli


class Particle:

    def __init__(self, id, ff_code, bound_min, bound_max, number_of_decimals, omega_generator, c1_generator,
                 c2_generator, mutation_mode, v_ones_max_percentage):
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
        self.real_dim = None
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

        self.real_dim = len(self.bound_min)
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
        if (self.mutation_mode == 1):
            self.mutation()

        return

    def mutation(self):
        v_ones_max_number = math.ceil(self.v_ones_max_percentage * self.bit_dim)
        positions_of_ones = list(self.v.findall('0b1', bytealigned=0))
        number_of_subs = len(positions_of_ones) - v_ones_max_number
        if (number_of_subs > 0):
            positions_of_subs = random.sample(positions_of_ones, number_of_subs)
            self.v.invert(positions_of_subs)
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
            sum_ = 0
            for i in range(d):
                xi = self.float_position[i]
                sum_ = sum_ + (xi ** 2 - 10 * math.cos(2 * math.pi * xi))
            self.fitness_value = 10 * d + sum_

        elif self.ff_code == 3:  # Ackley
            a = 20
            b = 0.2
            c = 2 * math.pi
            d = len(self.bound_min)
            sum1 = 0
            sum2 = 0
            for i in range(d):
                xi = self.float_position[i]
                sum1 += xi ** 2
                sum2 += math.cos(c * xi)
            term1 = -a * math.exp(-b * math.sqrt(sum1 / d))
            term2 = -math.exp(sum2 / d)

            self.fitness_value = term1 + term2 + a + math.exp(1)

        elif self.ff_code == 4:  # Rosenbrock
            d = len(self.bound_min)
            sum_ = 0
            for i in range(d - 1):
                new = 100 * (self.float_position[i + 1] - self.float_position[i] ** 2) ** 2 + (
                        self.float_position[i] - 1) ** 2
                sum_ = sum_ + new
            self.fitness_value = sum_

        elif self.ff_code == 5:  # Dropwave
            frac1 = 1 + math.cos(12 * math.sqrt(self.float_position[0] ** 2 + self.float_position[1] ** 2))
            frac2 = 0.5 * (self.float_position[0] ** 2 + self.float_position[1] ** 2) + 2

            self.fitness_value = -frac1 / frac2


        elif self.ff_code == 6:  # Bukin N.6
            term1 = 100 * math.sqrt(abs(self.float_position[1] - 0.01 * self.float_position[0] ** 2))
            term2 = 0.01 * abs(self.float_position[0] + 10)
            self.fitness_value = term1 + term2

        elif self.ff_code == 7:  # De Jong N.5
            x1 = self.float_position[0]
            x2 = self.float_position[1]
            s = 0

            A = np.zeros((2, 25))
            a = np.array([-32, -16, 0, 16, 32])
            A[0, :] = np.tile(a, 5)
            ar = np.tile(a, 5).reshape(5, 5).T.flatten()
            A[1, :] = ar

            for ii in range(25):
                a1i = A[0, ii]
                a2i = A[1, ii]
                term1 = ii + 1
                term2 = (x1 - a1i) ** 6
                term3 = (x2 - a2i) ** 6
                new = 1 / (term1 + term2 + term3)
                s += new

            self.fitness_value = 1 / (0.002 + s)

        elif self.ff_code == 8:  # Schwefel
            d = len(self.bound_min)
            sum_ = 0
            for i in range(d):
                sum_ = sum_ + self.float_position[i] * math.sin(math.sqrt(abs(self.float_position[i])))
            self.fitness_value = 418.9829 * d - sum_

        elif self.ff_code == 9:  # Cross-in-Tray
            fact1 = math.sin(self.float_position[0]) * math.sin(self.float_position[1])
            fact2 = math.exp(abs(100 - math.sqrt(self.float_position[0] ** 2 + self.float_position[1] ** 2) / math.pi))

            self.fitness_value = -0.0001 * (abs(fact1 * fact2) + 1) ** 0.1

        elif self.ff_code == 10:  # Holder Table
            fact1 = math.sin(self.float_position[0]) * math.cos(self.float_position[1])
            fact2 = math.exp(abs(1 - math.sqrt(self.float_position[0] ** 2 + self.float_position[1] ** 2) / math.pi))

            self.fitness_value = -abs(fact1 * fact2)

        elif self.ff_code == 11:  # Bohachevsky
            term1 = self.float_position[0] ** 2
            term2 = 2 * self.float_position[1] ** 2
            term3 = -0.3 * math.cos(3 * math.pi * self.float_position[0])
            term4 = -0.4 * math.cos(4 * math.pi * self.float_position[1])

            self.fitness_value = term1 + term2 + term3 + term4 + 0.7

        elif self.ff_code == 12:  # Griewank
            d = len(self.bound_min)
            sum_ = 0
            prod = 1

            for i in range(d):
                sum_ = sum_ + self.float_position[i] ** 2 / 4000
                prod = prod * math.cos(self.float_position[i] / math.sqrt(i + 1))

            self.fitness_value = sum_ - prod + 1

        elif self.ff_code == 13:  # Easom
            fact1 = -math.cos(self.float_position[0]) * math.cos(self.float_position[1])
            fact2 = math.exp(-(self.float_position[0] - math.pi) ** 2 - (self.float_position[1] - math.pi) ** 2)

            self.fitness_value = fact1 * fact2

        elif self.ff_code == 14:  # Dixon-Price
            d = len(self.bound_min)
            term1 = (self.float_position[0] - 1) ** 2
            sum_ = 0

            for i in range(1, d):
                new = i * (2 * self.float_position[i] ** 2 - self.float_position[i - 1]) ** 2
                sum_ += new

            self.fitness_value = term1 + sum_

        elif self.ff_code == 15:  # Six-Hump Camel
            term1 = (4 - 2.1 * self.float_position[0] ** 2 + (self.float_position[0] ** 4) / 3) * self.float_position[
                0] ** 2
            term2 = self.float_position[0] * self.float_position[1]
            term3 = (-4 + 4 * self.float_position[1] ** 2) * self.float_position[1] ** 2

            self.fitness_value = term1 + term2 + term3

        elif self.ff_code == 16:  # Gramacy & Lee Function
            term1 = math.sin(10 * math.pi * self.float_position[0]) / (2 * self.float_position[0])
            term2 = (self.float_position[0] - 1) ** 4

            self.fitness_value = term1 + term2

        elif self.ff_code == 17:  # Shubert
            sum1 = 0
            sum2 = 0

            for i in range(1, 6):
                new1 = i * math.cos((i + 1) * self.float_position[0] + i)
                new2 = i * math.cos((i + 1) * self.float_position[1] + i)
                sum1 += new1
                sum2 += new2

            self.fitness_value = sum1 * sum2

        elif self.ff_code == 18:  # Langermann
            m = 5
            d = 2
            c = np.array([1, 2, 5, 2, 3])
            a = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
            outer = 0
            for ii in range(m):
                inner = 0
                for jj in range(d):
                    xj = self.float_position[jj]
                    aij = a[ii, jj]
                    inner = inner + (xj - aij) ** 2
                new = c[ii] * np.exp(-inner / np.pi) * np.cos(np.pi * inner)
                outer = outer + new

            self.fitness_value = outer

        elif self.ff_code == 19:  # Schaffer N.2
            fact1 = (math.sin(self.float_position[0] ** 2 - self.float_position[1] ** 2)) ** 2 - 0.5
            fact2 = (1 + 0.001 * (self.float_position[0] ** 2 + self.float_position[1] ** 2)) ** 2

            self.fitness_value = 0.5 + fact1 / fact2

        elif self.ff_code == 20:  # Trid
            d = len(self.bound_min)
            sum1 = (self.float_position[0] - 1) ** 2
            sum2 = 0

            for i in range(1, d):
                sum1 = sum1 + (self.float_position[i] - 1) ** 2
                sum2 = sum2 + self.float_position[i] * self.float_position[i - 1]

            self.fitness_value = sum1 - sum2


        elif self.ff_code == 21:  # Sphere
            d = len(self.bound_min)
            sum_ = 0
            for i in range(d):
                sum_ += self.float_position[i] ** 2

            self.fitness_value = sum_

        elif self.ff_code == 22:  # Sum Squares function
            d = len(self.bound_min)
            sum_ = 0
            for i in range(d):
                sum_ += i * self.float_position[i] ** 2

            self.fitness_value = sum_


        elif self.ff_code == 23:  # Sum of Different Powers function
            d = len(self.bound_min)
            sum_ = 0
            for i in range(d):
                sum_ += (abs(self.float_position[i])) ** (i + 1)

            self.fitness_value = sum_

        elif self.ff_code == 24:  # Rotated Hyper-Ellipsoid
            d = len(self.bound_min)
            outer = 0
            for i in range(d):
                inner = 0
                for j in range(i + 1):
                    inner += self.float_position[j] ** 2
                outer += inner
            self.fitness_value = outer

        elif self.ff_code == 25:  # Booth
            term1 = (self.float_position[0] + 2 * self.float_position[1] - 7) ** 2
            term2 = (2 * self.float_position[0] + self.float_position[1] - 5) ** 2

            self.fitness_value = term1 + term2

        elif self.ff_code == 26:  # Matyas
            term1 = 0.26 * (self.float_position[0] ** 2 + self.float_position[1] ** 2)
            term2 = -0.48 * self.float_position[0] * self.float_position[1]

            self.fitness_value = term1 + term2


        elif self.ff_code == 27:  # McCormick
            term1 = math.sin(self.float_position[0] + self.float_position[1])
            term2 = (self.float_position[0] - self.float_position[1]) ** 2
            term3 = -1.5 * self.float_position[0]
            term4 = 2.5 * self.float_position[1]

            self.fitness_value = term1 + term2 + term3 + term4 + 1


        elif self.ff_code == 28:  # Levy
            d = len(self.bound_min)
            w = np.empty(d)
            for i in range(d):
                w[i] = 1 + (self.float_position[i] - 1) / 4

            term1 = (math.sin(math.pi * w[0])) ** 2
            term3 = (w[d-1] - 1) ** 2 * (1 + (math.sin(2 * math.pi * w[d-1])) ** 2)

            sum_ = 0
            for i in range(d - 1):
                new = (w[i] - 1) ** 2 * (1 + 10 * (math.sin(math.pi * w[i] + 1)) ** 2)
                sum_ += new

            self.fitness_value = term1 + sum_ + term3

        elif self.ff_code == 29:  # Levy N.13

            term1 = (math.sin(3 * math.pi * self.float_position[0])) ** 2
            term2 = (self.float_position[0] - 1) ** 2 * (1 + (math.sin(3 * math.pi * self.float_position[1])) ** 2)
            term3 = (self.float_position[1] - 1) ** 2 * (1 + (math.sin(2 * math.pi * self.float_position[1])) ** 2)

            self.fitness_value = term1 + term2 + term3


        elif self.ff_code == 30:  # Schaffer N.4
            fact1 = (math.cos(math.sin(abs(self.float_position[0] ** 2 - self.float_position[1] ** 2)))) ** 2 - 0.5
            fact2 = (1 + 0.001 * (self.float_position[0] ** 2 + self.float_position[1] ** 2)) ** 2

            self.fitness_value = 0.5 + fact1 / fact2

        elif self.ff_code == 31:  # Power Sum function
            d=4
            b=[8, 18, 44, 114]
            outer = 0
            for i in range(d):
                inner = 0
                for j in range(d):
                    inner += self.float_position[i] ** (i+1)

                outer = outer + (inner - b[i]) ** 2

            self.fitness_value = outer

        elif self.ff_code == 32:  # Zakharov
            d = len(self.bound_min)
            sum1 = 0
            sum2 = 0
            for i in range(d):
                sum1 += self.float_position[i] ** 2
                sum2 += 0.5 * (i+1) * self.float_position[i]

            self.fitness_value = sum1 + sum2 ** 2 + sum2 ** 4

        elif self.ff_code ==33:  #Three Hump Camel
            term1 = 2 * self.float_position[0] ** 2
            term2 = -1.05 * self.float_position[0] ** 4
            term3 = self.float_position[0] ** 6 / 6
            term4 = self.float_position[0] * self.float_position[1]
            term5 = self.float_position[1] ** 2

            self.fitness_value = term1 + term2 + term3 + term4 + term5

        elif self.ff_code == 34: #Beale
            term1 = (1.5 - self.float_position[0] + self.float_position[0] * self.float_position[1]) ** 2;
            term2 = (2.25 - self.float_position[0] + self.float_position[0] * self.float_position[1] ** 2) ** 2
            term3 = (2.625 - self.float_position[0] + self.float_position[0] * self.float_position[1] ** 3) ** 2

            self.fitness_value = term1 + term2 + term3

        elif self.ff_code == 35: #Branin
            t = 1 / (8 * math.pi)
            s = 10
            r=6
            c=5/math.pi
            b=5.1/(4*math.pi**2)
            a=1

            term1 = a * (self.float_position[1] - b * self.float_position[0] ** 2 + c * self.float_position[0] - r) ** 2
            term2 = s * (1 - t) * math.cos(self.float_position[0])

            self.fitness_value = term1 + term2 + s


        elif self.ff_code == 36: #Michalewicz
            m=10
            d=len(self.bound_min)
            sum_=0

            for i in range(d):
                new=math.sin(self.float_position[i])* (math.sin((i+1)*self.float_position[i]**2/math.pi))**(2*m)
                sum_ +=new

            self.fitness_value=-sum_


        elif self.ff_code == 37: #Colville
            term1 = 100 * (self.float_position[0] ** 2 - self.float_position[1]) ** 2
            term2 = (self.float_position[0] - 1) ** 2
            term3 = (self.float_position[2] - 1) ** 2
            term4 = 90 * (self.float_position[2] ** 2 - self.float_position[3]) ** 2
            term5 = 10.1 * ((self.float_position[1] - 1) ** 2 + (self.float_position[3] - 1) ** 2)
            term6 = 19.8 * (self.float_position[1] - 1) * (self.float_position[3] - 1)

            self.fitness_value = term1 + term2 + term3 + term4 + term5 + term6







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
