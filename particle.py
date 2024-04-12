import random
import math
import numpy as np
from scipy.stats import bernoulli
import antenna_functions as antenna
from bitstring import BitArray


# bitstring is a Python module that makes the creation and analysis
# of binary data as simple and efficient as possible.
# Documentation https://bitstring.readthedocs.io/en/stable/index.html

class Particle:

    def __init__(self, id, ff_code, bound_min, bound_max, number_of_decimals, omega_generator, c1_generator,
                 c2_generator, mutation_mode, v_mutation_factor, electromagnetic_problem_mode, i_amplitude_array, n_x,
                 n_y, d_x, d_y, b_x, b_y, theta0, phi0, ff_in_parallel):

        # Particle attributes initialization
        self.id = id
        self.ff_code = ff_code
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.number_of_decimals = number_of_decimals

        self.omega_generator = omega_generator
        self.c1_generator = c1_generator
        self.c2_generator = c2_generator

        self.mutation_mode = mutation_mode
        self.v_mutation_factor = v_mutation_factor

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
        self.ff_in_parallel = ff_in_parallel

        self.fitness_value = None  # Fitness value of the particle
        self.fitness_value_p_best = np.inf  # Personal best fitness value, initialized to positive infinity

        self.bit_dim = 0  # Number of bits in the binary representation
        self.real_dim = len(self.bound_min)  # Dimensionality of the problem in the real space
        self.compute_bit_dim()  # Calculate the number of bits based on the specified number of decimals

        # x and v is BitArray objects .
        # They have been randomly initialized according to Bernoulli-0.5 distribution

        self.x = self.generate_random_bitstring(random_genarator=0.5)  # Current position

        self.v = self.generate_random_bitstring(random_genarator=0.5)  # Velocity

        self.test_bitstring_in_bounds()  # Ensure that the initial position is within bounds

        self.p_best = self.x  # p_best is a BitArray object representing the personal best position initialized to the current position

    def __str__(self):
        # Define the string representation of the Particle object.

        return f"id:{self.id}\tmf:{self.v_mutation_factor}\tomega={self.omega_generator}\tx:{self.x.bin}\tfloat_pos:{self.float_position}\t fitness_value:{self.fitness_value}" \
               f"\tpbest:{self.p_best.bin}\t fitness_value_p_best:{self.fitness_value_p_best}"

    def compute_bit_dim(self):
        # Compute the number of bits required to represent the range of each variable.
        for i in range(self.real_dim):
            # Compute the range of each variable
            float_space = self.bound_max[i] - self.bound_min[i]
            # Calculate the number of bits required to represent the range with the specified number of decimals
            self.bit_dim += math.floor(math.log2(float_space * math.pow(10, self.number_of_decimals))) + 1
        return

    def generate_random_bitstring(self, random_genarator):
        # Generate a random BitArray of the proper dimension according to Bernoulli distribution.
        key1 = ""
        # Generate random bits using Bernoulli distribution
        random_data = bernoulli.rvs(size=self.bit_dim, p=random_genarator)
        for i in range(self.bit_dim):
            temp = str(random_data[i])
            key1 += temp
        # Convert the list of bits into a BitArray object
        return BitArray(bin=key1)

    def test_bitstring_in_bounds(self):
        # Tests if the position bitstring corresponds to a valid float position for the selected fitness function
        # if not it changes the position bitstring to the limits of valid space
        # In addition computes and saves the float position of the current position bitstring

        # Compute the float position of the current position bitstring
        self.float_position = self.driver_bin_to_float(self.x)
        flag = 0
        for i in range(self.real_dim):
            # Check if the float position is within bounds
            if self.float_position[i] > self.bound_max[i]:
                self.float_position[i] = self.bound_max[i]
                flag = 1
            elif self.float_position[i] < self.bound_min[i]:
                self.float_position[i] = self.bound_min[i]
                flag = 1
                print("alarm 1")
        # If the float position is out of bounds, adjust the position bitstring accordingly
        if flag == 1:
            self.x = self.driver_float_to_bin(self.float_position)
        return

    def driver_bin_to_float(self, input_bitstring):
        # Convert a binary bitstring to a float array.
        # The conversion is based on the provided bounds and number of decimals.
        # Returns a float array corresponding to the input binary bitstring.

        if self.electromagnetic_problem_mode == 0:
            output_float_array = np.zeros(self.real_dim)
        # If the software works on the array thinning problem the float array is converted in np.int32 array in order
        # to be compatible with C methods
        elif self.electromagnetic_problem_mode == 1:
            output_float_array = np.zeros(self.real_dim, dtype=np.int32)
        string_counter = 0
        for i in range(self.real_dim):
            # Compute the range of the current variable
            float_space = self.bound_max[i] - self.bound_min[i]
            # Calculate the number of bits needed to represent the range with the specified number of decimals
            number_of_bits_needed = math.floor(math.log2(float_space * math.pow(10, self.number_of_decimals))) + 1
            if number_of_bits_needed < 1:
                print("Error in driver bin to float . Impossible number of needed bits")
            # Extract the bits corresponding to the current variable
            output = input_bitstring.bin[string_counter:string_counter + number_of_bits_needed]
            string_counter += number_of_bits_needed
            # Convert the binary string to an integer and then to a float
            output = int(output, base=2)
            output = output / math.pow(10, self.number_of_decimals)
            output = output + self.bound_min[i]
            output_float_array[i] = output
        return output_float_array

    def driver_float_to_bin(self, float_number_array):
        # Convert a float array to a binary bitstring.
        # The conversion is based on the provided bounds and number of decimals.
        # Returns a BitArray representing the binary bitstring corresponding to the input float array.

        output_bitstring = BitArray()
        for i in range(len(float_number_array)):
            # Compute the range of the current variable
            float_space = self.bound_max[i] - self.bound_min[i]
            # Calculate the number of bits needed to represent the range with the specified number of decimals
            number_of_bits_needed = math.floor(math.log2(float_space * math.pow(10, self.number_of_decimals))) + 1
            # Normalize the float number to fit within the bit representation
            float_number = float_number_array[i] - self.bound_min[i]
            float_number = int(float_number * math.pow(10, self.number_of_decimals))
            # Append the binary representation of the normalized float number to the output bitstring
            output_bitstring.append(BitArray(uint=float_number, length=number_of_bits_needed))
        return output_bitstring

    # Calculates new velocity

    def upgrade_vel(self, g_best):
        # Update the velocity of the particle based on the global best position.
        # - g_best: Global best position

        # Update the velocity using the PSO formula
        self.v = (self.generate_random_bitstring(self.omega_generator) & self.v) | (
                ((self.x ^ g_best) & self.generate_random_bitstring(self.c2_generator)) | (
                (self.x ^ self.p_best) & self.generate_random_bitstring(self.c1_generator)))

        # If mutation mode is enabled, apply mutation
        if self.mutation_mode == 1:
            self.mutation()
        return

    def mutation(self):
        # Apply mutation to the velocity bitstring.
        # The mutation is based on the value of self.v_mutation_factor.
        # If '0b0' is used, mutation is based on 0s in the velocity bitstring.(0->1)
        # If '0b1' is used, mutation is based on 1s in the velocity bitstring.(1->0)

        # Determine the maximum number of 1s to be mutated based on the mutation factor
        v_ones_max_number = math.ceil(self.v_mutation_factor * self.bit_dim)
        # Find the positions of zeros in the velocity bitstring
        positions_of_zeros = list(self.v.findall('0b0', bytealigned=False))
        # Calculate the number of substitutions needed
        number_of_subs = len(positions_of_zeros) - v_ones_max_number
        # If the number of substitutions is positive, perform mutation
        if number_of_subs > 0:
            # Randomly select positions to perform mutations
            positions_of_subs = random.sample(positions_of_zeros, number_of_subs)
            # Invert the selected positions to perform mutations
            self.v.invert(positions_of_subs)
        return

    def upgrade_position(self):
        # Update the position using velocity
        self.x = self.x ^ self.v
        # Ensure the updated position is within bounds
        self.test_bitstring_in_bounds()
        return

    def upgrade_pbest(self):
        # If the current fitness value is better than the personal best fitness value, update p_best and fitness_value_p_best
        if self.fitness_value < self.fitness_value_p_best:
            self.p_best = self.x
            self.fitness_value_p_best = self.fitness_value
        return

    def fitness_function(self):  # Electromagnetic Problem
        # Depending on the value of ff_code computes the value of the fitness function that correspond to the current
        # position of the particle
        if self.ff_code == 0:
            directivity = antenna.directivity_planar_array_antenna_c(active_elements_array=self.float_position,
                                                                     i_amplitude_array=self.i_amplitude_array,
                                                                     d_x=self.d_x, d_y=self.d_y, b_x=self.b_x,
                                                                     b_y=self.b_y, n_x=self.n_x, n_y=self.n_y,
                                                                     theta0=self.theta0, phi0=self.phi0)

            sll = antenna.sll_planar_array_antenna_c(active_elements_array=self.float_position,
                                                     i_amplitude_array=self.i_amplitude_array,
                                                     d_x=self.d_x, d_y=self.d_y, b_x=self.b_x,
                                                     b_y=self.b_y, n_x=self.n_x, n_y=self.n_y
                                                     )

            self.fitness_value = -0.3 * directivity + 0.7 * max(0, sll + 20)

            # if parallel computation of particle fitness values is enabled we need to return
            # the fitness value to extract them from parallel pool
            if self.ff_in_parallel == True:
                return self.fitness_value

        if self.ff_code == 1:  # Ackley
            a = 20
            b = 0.2
            c = 2 * math.pi
            d = self.real_dim
            sum1 = 0
            sum2 = 0
            for i in range(d):
                xi = self.float_position[i]
                sum1 += xi ** 2
                sum2 += math.cos(c * xi)
            term1 = -a * math.exp(-b * math.sqrt(sum1 / d))
            term2 = -math.exp(sum2 / d)

            self.fitness_value = term1 + term2 + a + math.exp(1)

        elif self.ff_code == 2:  # Beale
            term1 = (1.5 - self.float_position[0] + self.float_position[0] * self.float_position[1]) ** 2
            term2 = (2.25 - self.float_position[0] + self.float_position[0] * self.float_position[1] ** 2) ** 2
            term3 = (2.625 - self.float_position[0] + self.float_position[0] * self.float_position[1] ** 3) ** 2

            self.fitness_value = term1 + term2 + term3

        elif self.ff_code == 3:  # Bohachevsky
            term1 = self.float_position[0] ** 2
            term2 = 2 * self.float_position[1] ** 2
            term3 = -0.3 * math.cos(3 * math.pi * self.float_position[0])
            term4 = -0.4 * math.cos(4 * math.pi * self.float_position[1])

            self.fitness_value = term1 + term2 + term3 + term4 + 0.7

        elif self.ff_code == 4:  # Booth
            term1 = (self.float_position[0] + 2 * self.float_position[1] - 7) ** 2
            term2 = (2 * self.float_position[0] + self.float_position[1] - 5) ** 2

            self.fitness_value = term1 + term2

        elif self.ff_code == 5:  # Branin
            t = 1 / (8 * math.pi)
            s = 10
            r = 6
            c = 5 / math.pi
            b = 5.1 / (4 * math.pi ** 2)
            a = 1

            term1 = a * (self.float_position[1] - b * self.float_position[0] ** 2 + c * self.float_position[0] - r) ** 2
            term2 = s * (1 - t) * math.cos(self.float_position[0])

            self.fitness_value = term1 + term2 + s

        elif self.ff_code == 6:  # Bukin N.6
            term1 = 100 * math.sqrt(abs(self.float_position[1] - 0.01 * self.float_position[0] ** 2))
            term2 = 0.01 * abs(self.float_position[0] + 10)
            self.fitness_value = term1 + term2

        elif self.ff_code == 7:  # Colville
            term1 = 100 * (self.float_position[0] ** 2 - self.float_position[1]) ** 2
            term2 = (self.float_position[0] - 1) ** 2
            term3 = (self.float_position[2] - 1) ** 2
            term4 = 90 * (self.float_position[2] ** 2 - self.float_position[3]) ** 2
            term5 = 10.1 * ((self.float_position[1] - 1) ** 2 + (self.float_position[3] - 1) ** 2)
            term6 = 19.8 * (self.float_position[1] - 1) * (self.float_position[3] - 1)

            self.fitness_value = term1 + term2 + term3 + term4 + term5 + term6

        elif self.ff_code == 8:  # Cross-in-Tray
            fact1 = math.sin(self.float_position[0]) * math.sin(self.float_position[1])
            fact2 = math.exp(abs(100 - math.sqrt(self.float_position[0] ** 2 + self.float_position[1] ** 2) / math.pi))

            self.fitness_value = -0.0001 * (abs(fact1 * fact2) + 1) ** 0.1

        elif self.ff_code == 9:  # De Jong N.5
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

        elif self.ff_code == 10:  # Dixon-Price
            d = self.real_dim
            term1 = (self.float_position[0] - 1) ** 2
            sum_ = 0

            for i in range(1, d):
                new = i * (2 * self.float_position[i] ** 2 - self.float_position[i - 1]) ** 2
                sum_ += new

            self.fitness_value = term1 + sum_

        elif self.ff_code == 11:  # Dropwave
            frac1 = 1 + math.cos(12 * math.sqrt(self.float_position[0] ** 2 + self.float_position[1] ** 2))
            frac2 = 0.5 * (self.float_position[0] ** 2 + self.float_position[1] ** 2) + 2

            self.fitness_value = -frac1 / frac2

        elif self.ff_code == 12:  # Easom
            fact1 = -math.cos(self.float_position[0]) * math.cos(self.float_position[1])
            fact2 = math.exp(-(self.float_position[0] - math.pi) ** 2 - (self.float_position[1] - math.pi) ** 2)

            self.fitness_value = fact1 * fact2

        elif self.ff_code == 13:  # Eggholder
            a = math.sqrt(math.fabs(self.float_position[1] + self.float_position[0] / 2 + 47))
            b = math.sqrt(math.fabs(self.float_position[0] - (self.float_position[1] + 47)))
            self.fitness_value = -(self.float_position[1] + 47) * math.sin(a) - self.float_position[0] * math.sin(b)

        elif self.ff_code == 14:  # Forrester
            fact1 = (6 * self.float_position[0] - 2) ** 2
            fact2 = math.sin(12 * self.float_position[0] - 4)

            self.fitness_value = fact1 * fact2

        elif self.ff_code == 15:  # Goldstein-Price
            fact1a = (self.float_position[0] + self.float_position[1] + 1) ** 2
            fact1b = 19 - 14 * self.float_position[0] + 3 * self.float_position[0] ** 2 - 14 * self.float_position[
                1] + 6 * self.float_position[0] * self.float_position[1] + 3 * self.float_position[1] ** 2
            fact1 = 1 + fact1a * fact1b

            fact2a = (2 * self.float_position[0] - 3 * self.float_position[1]) ** 2
            fact2b = 18 - 32 * self.float_position[0] + 12 * self.float_position[0] ** 2 + 48 * self.float_position[
                1] - 36 * self.float_position[0] * self.float_position[1] + 27 * self.float_position[1] ** 2
            fact2 = 30 + fact2a * fact2b

            self.fitness_value = fact1 * fact2

        elif self.ff_code == 16:  # Gramacy & Lee Function
            term1 = math.sin(10 * math.pi * self.float_position[0]) / (2 * self.float_position[0])
            term2 = (self.float_position[0] - 1) ** 4

            self.fitness_value = term1 + term2

        elif self.ff_code == 17:  # Griewank
            d = self.real_dim
            sum_ = 0
            prod = 1

            for i in range(d):
                sum_ = sum_ + self.float_position[i] ** 2 / 4000
                prod = prod * math.cos(self.float_position[i] / math.sqrt(i + 1))

            self.fitness_value = sum_ - prod + 1

        elif self.ff_code == 18:  # Hartmann 3-D
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            a = np.array([[3.0, 10, 30],
                          [0.1, 10, 35],
                          [3.0, 10, 30],
                          [0.1, 10, 35]])
            p = 10 ** (-4) * np.array([[3689, 1170, 2673],
                                       [4699, 4387, 7470],
                                       [1091, 8732, 5547],
                                       [381, 5743, 8828]])

            outer = 0
            for i in range(4):
                inner = 0
                for j in range(3):
                    inner += a[i, j] * (self.float_position[j] - p[i, j]) ** 2
                outer += alpha[i] * np.exp(-inner)
            self.fitness_value = -outer

        elif self.ff_code == 19:  # Hartmann 4-D
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            a = np.array([[10, 3, 17, 3.5, 1.7, 8],
                          [0.05, 10, 17, 0.1, 8, 14],
                          [3, 3.5, 1.7, 10, 17, 8],
                          [17, 8, 0.05, 10, 0.1, 14]])
            p = 10 ** (-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                       [2329, 4135, 8307, 3736, 1004, 9991],
                                       [2348, 1451, 3522, 2883, 3047, 6650],
                                       [4047, 8828, 8732, 5743, 1091, 381]])
            outer = 0
            for i in range(4):
                inner = 0
                for j in range(4):
                    inner += a[i, j] * (self.float_position[j] - p[i, j]) ** 2
                outer += alpha[i] * np.exp(-inner)
            self.fitness_value = (1.1 - outer) / 0.839

        elif self.ff_code == 20:  # Hartmann 6-D

            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            a = np.array([[10, 3, 17, 3.5, 1.7, 8],
                          [0.05, 10, 17, 0.1, 8, 14],
                          [3, 3.5, 1.7, 10, 17, 8],
                          [17, 8, 0.05, 10, 0.1, 14]])
            p = 10 ** (-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                       [2329, 4135, 8307, 3736, 1004, 9991],
                                       [2348, 1451, 3522, 2883, 3047, 6650],
                                       [4047, 8828, 8732, 5743, 1091, 381]])

            outer = 0
            for i in range(4):
                inner = 0
                for j in range(6):
                    inner += a[i, j] * (self.float_position[j] - p[i, j]) ** 2
                outer += alpha[i] * np.exp(-inner)
            self.fitness_value = -outer

        elif self.ff_code == 21:  # Holder Table
            fact1 = math.sin(self.float_position[0]) * math.cos(self.float_position[1])
            fact2 = math.exp(abs(1 - math.sqrt(self.float_position[0] ** 2 + self.float_position[1] ** 2) / math.pi))

            self.fitness_value = -abs(fact1 * fact2)

        elif self.ff_code == 22:  # Langermann
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

        elif self.ff_code == 23:  # Levy
            d = self.real_dim
            w = np.empty(d)
            for i in range(d):
                w[i] = 1 + (self.float_position[i] - 1) / 4

            term1 = (math.sin(math.pi * w[0])) ** 2
            term3 = (w[d - 1] - 1) ** 2 * (1 + (math.sin(2 * math.pi * w[d - 1])) ** 2)

            sum_ = 0
            for i in range(d - 1):
                new = (w[i] - 1) ** 2 * (1 + 10 * (math.sin(math.pi * w[i] + 1)) ** 2)
                sum_ += new

            self.fitness_value = term1 + sum_ + term3

        elif self.ff_code == 24:  # Levy N.13

            term1 = (math.sin(3 * math.pi * self.float_position[0])) ** 2
            term2 = (self.float_position[0] - 1) ** 2 * (1 + (math.sin(3 * math.pi * self.float_position[1])) ** 2)
            term3 = (self.float_position[1] - 1) ** 2 * (1 + (math.sin(2 * math.pi * self.float_position[1])) ** 2)

            self.fitness_value = term1 + term2 + term3

        elif self.ff_code == 25:  # Matyas
            term1 = 0.26 * (self.float_position[0] ** 2 + self.float_position[1] ** 2)
            term2 = -0.48 * self.float_position[0] * self.float_position[1]

            self.fitness_value = term1 + term2

        elif self.ff_code == 26:  # McCormick
            term1 = math.sin(self.float_position[0] + self.float_position[1])
            term2 = (self.float_position[0] - self.float_position[1]) ** 2
            term3 = -1.5 * self.float_position[0]
            term4 = 2.5 * self.float_position[1]

            self.fitness_value = term1 + term2 + term3 + term4 + 1

        elif self.ff_code == 27:  # Michalewicz
            m = 10
            d = self.real_dim
            sum_ = 0

            for i in range(d):
                new = math.sin(self.float_position[i]) * (
                    math.sin((i + 1) * self.float_position[i] ** 2 / math.pi)) ** (2 * m)
                sum_ += new

            self.fitness_value = -sum_

        elif self.ff_code == 28:  # Perm 0,d,b
            b = 10
            d = self.real_dim
            outer = 0
            for i in range(d):
                inner = 0
                for j in range(d):
                    inner += ((j + 1) + b) * (self.float_position[j] ** (i + 1) - 1 / ((j + 1) ** (i + 1)))
                outer += inner ** 2
            self.fitness_value = outer

        elif self.ff_code == 29:  # Perm d,b
            b = 0.5
            d = self.real_dim
            outer = 0
            for i in range(d):
                inner = 0
                for j in range(d):
                    inner += ((j + 1) ** (i + 1) + b) * ((self.float_position[j] / (j + 1)) ** (i + 1) - 1)
                outer += inner ** 2
            self.fitness_value = outer

        elif self.ff_code == 30:  # Powell
            d = self.real_dim
            sum_ = 0

            for i in range(1, int(d / 4) + 1):
                term1 = (self.float_position[4 * i - 4] + 10 * self.float_position[4 * i - 3]) ** 2
                term2 = 5 * (self.float_position[4 * i - 2] - self.float_position[4 * i - 1]) ** 2
                term3 = (self.float_position[4 * i - 3] - 2 * self.float_position[4 * i - 2]) ** 4
                term4 = 10 * (self.float_position[4 * i - 4] - self.float_position[4 * i - 1]) ** 4
                sum_ += term1 + term2 + term3 + term4

            self.fitness_value = sum_

        elif self.ff_code == 31:  # Power Sum function
            d = 4
            b = [8, 18, 44, 114]
            outer = 0
            for i in range(d):
                inner = 0
                for j in range(d):
                    inner += self.float_position[i] ** (i + 1)

                outer = outer + (inner - b[i]) ** 2

            self.fitness_value = outer

        elif self.ff_code == 32:  # Rastring

            d = self.real_dim
            sum_ = 0
            for i in range(d):
                xi = self.float_position[i]
                sum_ = sum_ + (xi ** 2 - 10 * math.cos(2 * math.pi * xi))
            self.fitness_value = 10 * d + sum_

        elif self.ff_code == 33:  # Rosenbrock
            d = self.real_dim
            sum_ = 0
            for i in range(d - 1):
                new = 100 * (self.float_position[i + 1] - self.float_position[i] ** 2) ** 2 + (
                        self.float_position[i] - 1) ** 2
                sum_ = sum_ + new
            self.fitness_value = sum_

        elif self.ff_code == 34:  # Rotated Hyper-Ellipsoid
            d = self.real_dim
            outer = 0
            for i in range(d):
                inner = 0
                for j in range(i + 1):
                    inner += self.float_position[j] ** 2
                outer += inner
            self.fitness_value = outer

        elif self.ff_code == 35:  # Schaffer N.2
            fact1 = (math.sin(self.float_position[0] ** 2 - self.float_position[1] ** 2)) ** 2 - 0.5
            fact2 = (1 + 0.001 * (self.float_position[0] ** 2 + self.float_position[1] ** 2)) ** 2

            self.fitness_value = 0.5 + fact1 / fact2

        elif self.ff_code == 36:  # Schaffer N.4
            fact1 = (math.cos(math.sin(abs(self.float_position[0] ** 2 - self.float_position[1] ** 2)))) ** 2 - 0.5
            fact2 = (1 + 0.001 * (self.float_position[0] ** 2 + self.float_position[1] ** 2)) ** 2

            self.fitness_value = 0.5 + fact1 / fact2

        elif self.ff_code == 37:  # Schwefel
            d = self.real_dim
            sum_ = 0
            for i in range(d):
                sum_ = sum_ + self.float_position[i] * math.sin(math.sqrt(abs(self.float_position[i])))
            self.fitness_value = 418.9829 * d - sum_

        elif self.ff_code == 38:  # Shekel
            m = 10  # if it's changed we have to change the global_min_value too
            b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
            c = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                          [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                          [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                          [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])

            outer = 0
            for i in range(m):
                inner = 0
                for j in range(4):
                    inner += (self.float_position[j] - c[j][i]) ** 2
                outer += 1 / (inner + b[i])

            self.fitness_value = - outer

        elif self.ff_code == 39:  # Shubert
            sum1 = 0
            sum2 = 0

            for i in range(1, 6):
                new1 = i * math.cos((i + 1) * self.float_position[0] + i)
                new2 = i * math.cos((i + 1) * self.float_position[1] + i)
                sum1 += new1
                sum2 += new2

            self.fitness_value = sum1 * sum2

        elif self.ff_code == 40:  # Six-Hump Camel
            term1 = (4 - 2.1 * self.float_position[0] ** 2 + (self.float_position[0] ** 4) / 3) * self.float_position[
                0] ** 2
            term2 = self.float_position[0] * self.float_position[1]
            term3 = (-4 + 4 * self.float_position[1] ** 2) * self.float_position[1] ** 2

            self.fitness_value = term1 + term2 + term3

        elif self.ff_code == 41:  # Sphere
            d = self.real_dim
            sum_ = 0
            for i in range(d):
                sum_ += self.float_position[i] ** 2

            self.fitness_value = sum_

        elif self.ff_code == 42:  # Styblinski-Tang
            d = self.real_dim
            sum_ = 0
            for i in range(d):
                sum_ += self.float_position[i] ** 4 - 16 * self.float_position[i] ** 2 + 5 * self.float_position[i]

            self.fitness_value = sum_ / 2

        elif self.ff_code == 43:  # Sum Squares function
            d = self.real_dim
            sum_ = 0
            for i in range(d):
                sum_ += i * self.float_position[i] ** 2

            self.fitness_value = sum_

        elif self.ff_code == 44:  # Sum of Different Powers function
            d = self.real_dim
            sum_ = 0
            for i in range(d):
                sum_ += (abs(self.float_position[i])) ** (i + 1)

            self.fitness_value = sum_

        elif self.ff_code == 45:  # Three Hump Camel
            term1 = 2 * self.float_position[0] ** 2
            term2 = -1.05 * self.float_position[0] ** 4
            term3 = self.float_position[0] ** 6 / 6
            term4 = self.float_position[0] * self.float_position[1]
            term5 = self.float_position[1] ** 2

            self.fitness_value = term1 + term2 + term3 + term4 + term5

        elif self.ff_code == 46:  # Trid
            d = self.real_dim
            sum1 = (self.float_position[0] - 1) ** 2
            sum2 = 0

            for i in range(1, d):
                sum1 = sum1 + (self.float_position[i] - 1) ** 2
                sum2 = sum2 + self.float_position[i] * self.float_position[i - 1]

            self.fitness_value = sum1 - sum2

        elif self.ff_code == 47:  # Zakharov
            d = self.real_dim
            sum1 = 0
            sum2 = 0
            for i in range(d):
                sum1 += self.float_position[i] ** 2
                sum2 += 0.5 * (i + 1) * self.float_position[i]

            self.fitness_value = sum1 + sum2 ** 2 + sum2 ** 4

        return

###TEST CODE
#
#
# bound_min = [0] * n_x*n_y
# bound_max = [1] * n_x*n_y
#
# p = Particle(id=0,ff_code=0,bound_min=bound_min,bound_max=bound_max,number_of_decimals=0,omega_generator=0.5,c1_generator=0.5,c2_generator=0.5,mutation_mode=1,v_mutation_factor=0.4)
#
# # print(p.float_position)
# for i in range(100):
#     p.fitness_function()
#     p.upgrade_pbest()
#     print(p)
#     p.upgrade_vel(p.generate_random_bitstring(0.5))
#     p.upgrade_position()
#     i+=1
