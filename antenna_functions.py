import numpy as np
import matplotlib.pyplot as plt

# Code in comments detects if we are on a Mac .If so the code will enable interactive plots .
# Mostly useful for the 3D radiation plot


# import platform
#
# if platform.system() == 'Darwin':
#     import matplotlib as mpl
#
#     mpl.use('macosx')
import ctypes
import os

# Get the directory path of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Initialize library_path variable to store the path of the library file
library_path = None

# Search for the library file within the script's directory and its subdirectories
for root, dirs, files in os.walk(script_directory):
    for file in files:
        if file.endswith('.so') and 'arrayFactor' in file:
            library_path = os.path.join(root, file)
            break

# Load the library if found
if library_path:
    try:
        array_factor_lib = ctypes.CDLL(library_path)  # Load the library using ctypes
        # print(f"Library loaded successfully from: {library_path}")    # Optional: Print the path of the loaded library
    except OSError as e:
        print(f"Error loading library: {e}")
else:
    print("Library not found in the script's directory.")

# Code for manual input of the library when the exact path is known
# try:
#     array_factor_lib = ctypes.CDLL('/home/n/nikopole/thesis/bpso/bpso_algorithm_6/arrayFactor.so')
#     array_factor_lib = ctypes.CDLL('arrayFactor.so')
# except OSError as e:
#     print(f"Error loading library: {e}")

# Declare functions c-signature

array_factor_lib.array_factor.restype = ctypes.c_double
array_factor_lib.array_factor.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # active_elements_array
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # i_amplitude_array
    ctypes.c_double, ctypes.c_double,  # d_x, d_y
    ctypes.c_double, ctypes.c_double,  # b_x, b_y
    ctypes.c_int, ctypes.c_int,  # n_x, n_y
    ctypes.c_double, ctypes.c_double]  # theta, phi

array_factor_lib.directivity_planar_array_antenna.restype = ctypes.c_double
array_factor_lib.directivity_planar_array_antenna.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # active_elements_array
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # i_amplitude_array
    ctypes.c_double, ctypes.c_double,  # d_x, d_y
    ctypes.c_double, ctypes.c_double,  # b_x, b_y
    ctypes.c_int, ctypes.c_int,  # n_x, n_y
    ctypes.c_double, ctypes.c_double  # theta0, phi0
]

array_factor_lib.sll_planar_array_antenna.restype = ctypes.c_double
array_factor_lib.sll_planar_array_antenna.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # active_elements_array
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # i_amplitude_array
    ctypes.c_double, ctypes.c_double,  # d_x, d_y
    ctypes.c_double, ctypes.c_double,  # b_x, b_y
    ctypes.c_int, ctypes.c_int,  # n_x, n_y
]


# Wrapper functions for easy use in Python
def array_factor_c(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y, theta, phi):
    return array_factor_lib.array_factor(active_elements_array, i_amplitude_array,
                                         d_x, d_y, b_x, b_y, n_x, n_y, theta, phi)


def directivity_planar_array_antenna_c(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y, theta0,
                                       phi0):
    return array_factor_lib.directivity_planar_array_antenna(active_elements_array, i_amplitude_array,
                                                             d_x, d_y, b_x, b_y, n_x, n_y, theta0, phi0)


def sll_planar_array_antenna_c(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y):
    return array_factor_lib.sll_planar_array_antenna(active_elements_array, i_amplitude_array,
                                                     d_x, d_y, b_x, b_y, n_x, n_y)


# Radiation patterns and sketch methods

# In var angle variable we put the strings: 'phi' or 'theta'.
# This will give us the phi-plane or theta-plane diagram respectively with the other angle stable at stable_angle _value
# stable_angle _value should be in rads
# file_name determines the name of the .png file that will be saved

def radiation_pattern_2d(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y, var_angle,
                         stable_angle_value, file_name):
    plt.figure(figsize=(8, 8))
    plt.axes(projection='polar')
    rads = np.arange(0, (2 * np.pi), 0.0001)  # specify the resolution for diagram
    for rad in rads:  # computes the array factor value
        if var_angle == "phi":
            r = abs(array_factor_c(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y,
                                   stable_angle_value, rad))
        elif var_angle == "theta":
            r = abs(array_factor_c(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y, rad,
                                   stable_angle_value))
        else:
            print("Wrong var_angle input .Accepted : 'theta' or 'phi'")  # error message
            return
        plt.polar(rad, r, 'g.')

    # Defining the title of the diagram

    if var_angle == "phi":
        plt.title(f"2D Polar Plot (theta={round(180 * stable_angle_value / np.pi, 2)} °)")  # theta in degrees
    elif var_angle == "theta":
        plt.title(f"2D Polar Plot (phi={round(180 * stable_angle_value / np.pi, 2)} °)")  # phi in degrees

    # Adjust layout and save the plot to a file
    plt.tight_layout()
    plt.savefig(f"{file_name}")
    # plt.show()       # Uncomment this line to display the plot interactively
    plt.clf()

    return


def radiation_pattern_3d(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y, file_name):
    # Specify the resolution for diagram
    n_points = 1000

    # Generate theta and phi values for the radiation pattern
    theta_vals = np.linspace(0, np.pi, n_points)
    phi_vals = np.linspace(0, 2 * np.pi, n_points)
    theta, phi = np.meshgrid(theta_vals, phi_vals)

    # Compute the array factor (AF) values for each combination of theta and phi
    af_values = np.zeros_like(theta)
    for i in range(n_points):
        for j in range(n_points):
            af_values[i, j] = np.abs(
                array_factor_c(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y,
                               theta[i, j], phi[i, j]))

    # Convert AF values to Cartesian coordinates for plotting
    x = af_values * np.sin(theta) * np.cos(phi)
    y = af_values * np.sin(theta) * np.sin(phi)
    z = af_values * np.cos(theta)

    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='k')  # You can choose a different colormap

    # Set labels and title for the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Radiation Pattern')

    # Set limits for better visualization
    mins = [np.min(x), np.min(y), np.min(z)]
    maxs = [np.max(x), np.max(y), np.max(z)]
    ax.set_xlim([np.min(mins), np.max(maxs)])
    ax.set_ylim([np.min(mins), np.max(maxs)])
    ax.set_zlim([np.min(mins), np.max(maxs)])

    # Set initial view angle
    ax.view_init(elev=30, azim=30)

    # Save the plot to a file
    plt.savefig(f"{file_name}", bbox_inches='tight')

    # Display the plot
    plt.show()

    # Clear the plot to avoid overlap if plotting multiple figures
    plt.clf()

    return


def planar_array_antenna_sketch(active_elements_array, d_x, d_y, n_x, n_y, file_name):
    # Initialize lists to store coordinates of active and inactive elements
    x_on = []
    y_on = []
    x_off = []
    y_off = []

    # Loop through each element in the array
    for j in range(n_y):
        for i in range(n_x):
            # Check if the current element is active (1) or inactive (0)
            if active_elements_array[j * n_x + i] == 1:
                x_on.append(i * d_x)  # Store x-coordinate of active element
                y_on.append(j * d_y)  # Store y-coordinate of active element
            elif active_elements_array[j * n_x + i] == 0:
                x_off.append(i * d_x)  # Store x-coordinate of inactive element
                y_off.append(j * d_y)  # Store y-coordinate of inactive element

    # Plot active and inactive elements
    plt.plot(x_on, y_on, 'o', color='g', label='on')  # Plot active elements in green
    plt.plot(x_off, y_off, 'o', color='r', label='off')  # Plot inactive elements in red

    # Set labels, title, and legend
    plt.xlabel("X-axis (λ)")
    plt.ylabel("Y-axis (λ)")
    plt.title("Planar array antenna structure")
    plt.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')  # Place legend outside the plot

    # Adjust aspect ratio to match the ratio of elements in each dimension
    plt.gca().set_aspect(n_x / n_y)

    # Add gridlines and set ticks
    plt.grid(True, which='both', linestyle='-', linewidth=2)  # Add major and minor gridlines
    plt.xticks([i * d_x for i in range(n_x)])  # Set x-axis ticks
    plt.yticks([j * d_y for j in range(n_y)])  # Set y-axis ticks

    # Adjust layout and save the plot to a file
    plt.tight_layout()
    plt.savefig(f"{file_name}")
    # plt.show()        # Uncomment this line to display the plot interactively

    # Clear the plot to avoid overlap if plotting multiple figures
    plt.clf()

    return

######Code to export diagrams given the bitstring of the antenna array
# nx = 16
# ny = 16
# active_ampl = np.array([1] * nx * ny, dtype=np.int32)
# dx = 1 / 2
# dy = 1 / 2
# # bx = 0
# # by = 0
# # theta0 = 0
# # phi0 = 0
# bx = -(3 * np.pi) / 4  # b_x b_y in rad
# by = -(np.sqrt(3) * np.pi) / 4
# theta0 = np.pi / 3
# phi0 = np.pi / 6
#
# binary_string = '1001011111110101000001110110010010000010111111000001101111110000100111111111010000110111111110111111111011111110110111111111010101101111111110111110011111110101000111111011111011111101111001101101011101111010100000111101010001011011110000010011100111011000'
# # # Convert binary string to NumPy array of int32
# active_el = np.array([int(bit) for bit in binary_string], dtype=np.int32)
#
# active_el_counter = 0
# for element in active_el:
#     if element == 1:
#         active_el_counter += 1
#
# print(f'{active_el_counter=}')
#
# radiation_pattern_2d(active_el, active_ampl, dx, dy, bx, by, nx, ny, 'phi', np.pi / 3, 'rad2d_phi')
# radiation_pattern_2d(active_el, active_ampl, dx, dy, bx, by, nx, ny, 'theta', np.pi / 6, 'rad2d_theta')
# radiation_pattern_3d(active_el, active_ampl, dx, dy, bx, by, nx, ny, "rad3d")
# planar_array_antenna_sketch(active_el, dx, dy, nx, ny, "sketch")
#
# directivity = directivity_planar_array_antenna_c(active_el, active_ampl, dx, dy, bx, by, nx, ny, theta0, phi0)
# sll = sll_planar_array_antenna_c(active_el, active_ampl, dx, dy, bx, by, nx, ny)
#
# print(f"f{directivity=}")
# print(f"{sll=}")
