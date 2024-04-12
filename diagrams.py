import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from results import Results
from parameters import Parameters
from antenna_functions import radiation_pattern_2d, radiation_pattern_3d, planar_array_antenna_sketch


def test_functions_histogram(data_array, param_dictionary):
    # Function to create plots when we optimize test functions

    # Extracting function names and accuracy percentages from the data array
    names = [row[0] for row in data_array]
    acc_perc = [row[9] for row in data_array]
    # Reverse the lists to display in descending order(alphabetically)
    names = names[::-1]
    acc_perc = acc_perc[::-1]
    # Assign colors based on accuracy percentage ranges
    colors = ['red' if x < 25 else 'orange' if x < 75 else 'green' for x in acc_perc]
    # Create a horizontal bar plot
    fig, ax = plt.subplots(figsize=(15, 15), gridspec_kw={'width_ratios': [1]})

    bars = ax.barh(names, acc_perc, color=colors)
    # Set labels and title
    ax.set_ylabel("Fitness Function Name")
    ax.set_xlabel("BPSO Accuracy %")
    ax.set_title(f"Fitness Function Results \n {param_dictionary}")
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('test_function_accuracy.png')
    plt.clf()  # Clear the current figure to avoid overlapping plots

    return


def parameter_tuning_plot(data_array, ff_name_array, parameters: Parameters, grouped_by: str, file_name,
                          firs_attribute_index,
                          second_attribute_index=None):

    # Function to create plots when parameter selection mode is on

    # Create a DataFrame from the data array
    if second_attribute_index is None:
        df = pd.DataFrame({'x': [row[-2] for row in data_array],
                           'y': [row[-1] for row in data_array],
                           'z': [row[firs_attribute_index] for row in data_array]})
        # Group data by the first attribute index
        groups = df.groupby('z')
        plt.figure(figsize=(14, 12))
        # Plot each group with a unique marker style
        for name, group in groups:
            plt.plot(group.x, group.y, marker='o', linestyle='', markersize=8, label=name)

    else:
        df = pd.DataFrame({'x': [row[-2] for row in data_array],
                           'y': [row[-1] for row in data_array],
                           'z': [row[firs_attribute_index] for row in data_array],
                           'marker_attribute': [row[second_attribute_index] for row in data_array]})

        # Group data by the first and second attribute indices
        groups = df.groupby(['z', 'marker_attribute'])

        plt.figure(figsize=(14, 12))

        # Define marker styles for each unique marker attribute
        marker_dict = {value: marker for value, marker in
                       zip(df['marker_attribute'].unique(), ['o', 's', '^', 'D', 'v'])}

        # Plot each group with a unique marker style
        for (z, marker_attribute), group in groups:
            marker = marker_dict[marker_attribute]
            plt.plot(group.x, group.y, marker=marker, linestyle='', markersize=8, label=f'{z}, {marker_attribute}')

    # Add legend and labels
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Accuracy %')
    plt.title(
        f'Parameters Selection Results \n '
        f'Function: {ff_name_array[parameters.ff_code]} \n'
        f'Mutation Mode = {parameters.mutation_mode}-Dynamic Mutation Mode = {parameters.dynamic_mutation_mode}-Dynamic Omega Mode = {parameters.dynamic_omega_mode}-Grouped by {grouped_by} ')
    # Save the plot and clear the figure
    plt.savefig(f'{file_name}.png')
    plt.clf()

    return


def simple_diagram(x_axis, y_axis, x_label, y_label, title):
    # Create simple dot plot (for directivity - sll values)
    plt.plot(x_axis, y_axis, 'o')
    plt.xlabel(f"{x_label}")
    plt.ylabel(f"{y_label}")
    plt.title(f"{title}")
    plt.tight_layout()
    plt.savefig(f"{title}")
    plt.clf()

    return


def diagrams(parameters: Parameters, results: Results, param_dictionary, ff_name_array):
    # Generate various diagrams based on the optimization process

    if parameters.parameters_selection_mode == 0 and parameters.electromagnetic_problem_mode == 1:
        # Generate a simple diagram showing Directivity and SLL for the result designs
        simple_diagram(results.directivity, results.sll, "Directivity", "SLL", "directivity_sll_plot")

        # Sort the data array according to fitness function value
        results.data_array.sort(key=lambda xx: (xx[2]), reverse=False)

        # Determine theta angle for theta radiation patterns if theta is 0 radiation pattern doesn't give any information
        theta_angle = parameters.theta0
        if parameters.theta0 == 0:
            theta_angle = np.pi / 2

        # Generate radiation pattern plot and sketch for the best design (index 0 of the sorted array )
        radiation_pattern_2d(active_elements_array=results.data_array[0][5],
                             i_amplitude_array=parameters.i_amplitude_array, d_x=parameters.d_x, d_y=parameters.d_y,
                             b_x=parameters.b_x, b_y=parameters.b_y,
                             n_x=parameters.n_x, n_y=parameters.n_y, var_angle='phi', stable_angle_value=theta_angle,
                             file_name=f"2Dplot_theta")
        radiation_pattern_2d(active_elements_array=results.data_array[0][5],
                             i_amplitude_array=parameters.i_amplitude_array, d_x=parameters.d_x, d_y=parameters.d_y,
                             b_x=parameters.b_x, b_y=parameters.b_y,
                             n_x=parameters.n_x, n_y=parameters.n_y, var_angle='theta',
                             stable_angle_value=parameters.phi0,
                             file_name=f"2Dplot_phi")
        radiation_pattern_3d(active_elements_array=results.data_array[0][5],
                             i_amplitude_array=parameters.i_amplitude_array, d_x=parameters.d_x, d_y=parameters.d_y,
                             b_x=parameters.b_x, b_y=parameters.b_y,
                             n_x=parameters.n_x, n_y=parameters.n_y, file_name=f"3Dplot")
        planar_array_antenna_sketch(active_elements_array=results.data_array[0][5], d_x=parameters.d_x,
                                    d_y=parameters.d_y,
                                    n_x=parameters.n_x, n_y=parameters.n_y, file_name=f"sketch")


    elif parameters.parameters_selection_mode == 0 and parameters.electromagnetic_problem_mode == 0:
        # Generate histogram for test functions
        test_functions_histogram(results.data_array, param_dictionary)

    elif parameters.parameters_selection_mode == 1 and parameters.electromagnetic_problem_mode == 0:
        # Plot parameter tuning based on different modes combination
        if parameters.mutation_mode == 0 and parameters.dynamic_omega_mode == 0:
            parameter_tuning_plot(results.data_array, ff_name_array, parameters, "omega",
                                  "parameters_tuning_plot_omega", 1)

        elif parameters.mutation_mode == 0 and parameters.dynamic_omega_mode != 0:
            parameter_tuning_plot(results.data_array, ff_name_array, parameters, "omega",
                                  "parameters_tuning_plot_omega", 1, 2)

        elif parameters.dynamic_mutation_mode == 0 and parameters.dynamic_omega_mode == 0:
            parameter_tuning_plot(results.data_array, ff_name_array, parameters, "mutation",
                                  "parameters_tuning_plot_mutation",
                                  1)
            parameter_tuning_plot(results.data_array, ff_name_array, parameters, "omega",
                                  "parameters_tuning_plot_omega", 2)

        elif parameters.dynamic_mutation_mode != 0 and parameters.dynamic_omega_mode == 0:
            parameter_tuning_plot(results.data_array, ff_name_array, parameters, "mutation",
                                  "parameters_tuning_plot_mutation_pairs", 1, 2)

        elif parameters.dynamic_mutation_mode == 0 and parameters.dynamic_omega_mode != 0:
            parameter_tuning_plot(results.data_array, ff_name_array, parameters, "omega",
                                  "parameters_tuning_plot_omega_pairs", 2, 3)

        elif parameters.dynamic_mutation_mode != 0 and parameters.dynamic_omega_mode != 0:
            parameter_tuning_plot(results.data_array, ff_name_array, parameters, "mutation",
                                  "parameters_tuning_plot_mutation_pairs", 1, 2)
            parameter_tuning_plot(results.data_array, ff_name_array, parameters, "omega",
                                  "parameters_tuning_plot_omega_pairs", 3, 4)

    return
