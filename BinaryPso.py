from Swarm import Swarm


class BinaryPso:
    def __init__(self, size, ff_code, number_of_decimals, omega_generator, c1_generator, c2_generator,
                 algorithm_iterations, file):
        self.size = size
        self.ff_code = ff_code
        self.number_of_decimals = number_of_decimals

        self.omega_generator = omega_generator
        self.c1_generator = c1_generator
        self.c2_generator = c2_generator

        self.algorithm_iterations = algorithm_iterations

        self.swarm = Swarm(size=self.size, ff_code=self.ff_code, number_of_decimals=self.number_of_decimals,
                           omega_generator=self.omega_generator, c1_generator=self.c1_generator,
                           c2_generator=self.c2_generator)

        for i in range(self.algorithm_iterations):
            self.swarm.evaluate_fitness_swarm()
            self.swarm.upgrade_pbest_swarm()
            self.swarm.upgrade_gbest()
            # print(self.swarm)
            self.swarm.upgrade_vel_swarm()
            self.swarm.upgrade_position_swarm()

        file.write(
            f"best is {self.swarm.gbest_value} at {self.swarm.g_best.bin} or {self.swarm.g_best_float_position}\n")
