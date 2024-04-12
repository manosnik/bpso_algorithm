#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <stdbool.h>

//C functions that implement the computation of the absolute value of array factor ,the directivity in given direction
//and the SLL

double  array_factor(int *active_elements_array, int *i_amplitude_array, double d_x, double d_y, double b_x, double b_y, int n_x, int n_y, double theta, double phi) {
    double complex af = 0;
    for (int j = 0; j < n_y; j++) {
        double complex af_x = 0;
        for (int i = 0; i < n_x; i++) {
            af_x += active_elements_array[j * n_x + i] * i_amplitude_array[j * n_x + i] * cexp(I * (i) * (
                    2 * M_PI * d_x * sin(theta) * cos(phi) + b_x));
        }
        af += af_x * cexp(I * (j) * (2 * M_PI * d_y * sin(theta) * sin(phi) + b_y));
    }
    af = cabs(af);
    return af;
}

double directivity_planar_array_antenna(int *active_elements_array, int *i_amplitude_array, double d_x, double d_y, double b_x, double b_y, int n_x, int n_y, double theta0, double phi0) {
    double term1 = 4 * M_PI * pow(array_factor(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y, theta0, phi0),2);
    // printf("Term 1 =%.2f \n",term1);
    double term2 = 0;
    double delta_theta = M_PI / 90;
    double delta_phi = M_PI / 90;
    for (double theta = 0; theta < M_PI/2; theta += delta_theta) {
        for (double phi = 0; phi < 2 * M_PI; phi += delta_phi) {
            term2 +=  pow(array_factor(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y, theta, phi),2) * sin(theta) * delta_theta * delta_phi;
        }
    }
    term2=2*term2;
    // printf("Term 2 =%.2f \n",term2);
    
    double directivity = 10 * log10(term1 / term2);
    return directivity;
}


double sll_planar_array_antenna(int *active_elements_array, int *i_amplitude_array, double d_x, double d_y, double b_x, double b_y, int n_x, int n_y){
    double delta_theta = M_PI / 360;
    double delta_phi = M_PI / 360;

    bool flag_theta_zero=false;
    bool flag_theta_pi=false;

    double main_lobe[3];
    main_lobe[0]=-10;

    double secondary_lobe[3];
    secondary_lobe[0]=-10;

    double previous_af;
    double current_af;
    double next_af;

    double previous_af_phi;
    double next_af_phi;

    double af_theta_0=array_factor(active_elements_array,i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y,0,0);

    for (double phi = 0; phi < 2*M_PI; phi += delta_phi) {
        for (double theta = 0; theta < M_PI/2; theta += delta_theta) {
        
        if (theta==0)
        {
            previous_af=array_factor(active_elements_array,i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y,theta+delta_theta,phi+M_PI);
            current_af=af_theta_0;
        }
        if (theta==M_PI)
        {
            next_af=array_factor(active_elements_array,i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y,theta-delta_theta,phi+M_PI);
        }
        else
        {
            next_af=array_factor(active_elements_array,i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y,theta+delta_theta,phi);
        }
        
        
        if (current_af > previous_af && current_af > next_af && (main_lobe[0] < 0 || current_af > main_lobe[0])) {
            if (theta == 0 && phi == 0 && flag_theta_zero == false) {
                if (main_lobe[0] > 0) {
                    secondary_lobe[0] = main_lobe[0];
                    secondary_lobe[1] = main_lobe[1];
                    secondary_lobe[2] = main_lobe[2];

                }
                main_lobe[0] = current_af;
                main_lobe[1] = theta;
                main_lobe[2] = phi;
                flag_theta_zero = true;
            }
            else if (theta == M_PI && phi == 0 && flag_theta_pi == false)
            {
                if (main_lobe[0] > 0) {
                    secondary_lobe[0] = main_lobe[0];
                    secondary_lobe[1] = main_lobe[1];
                    secondary_lobe[2] = main_lobe[2];

                }
                main_lobe[0] = current_af;
                main_lobe[1] = theta;
                main_lobe[2] = phi;
                flag_theta_pi = true;
            }
            else if ((theta == 0 && flag_theta_zero == true) || (theta == M_PI && flag_theta_pi == true))
            {
                previous_af=current_af;
                current_af=next_af;
                continue;

            }
            else{
                previous_af_phi=array_factor(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y, theta, phi - delta_phi);
                next_af_phi=array_factor(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y, theta, phi + delta_phi);
                if (current_af > previous_af_phi && current_af > next_af_phi) {
                    if (main_lobe[0] > 0) {
                        secondary_lobe[0] = main_lobe[0];
                        secondary_lobe[1] = main_lobe[1];
                        secondary_lobe[2] = main_lobe[2];

                    }
                    main_lobe[0] = current_af;
                    main_lobe[1] = theta;
                    main_lobe[2] = phi;
                }

            
        }
        
        
            
        
        }
        else if (current_af > previous_af && current_af > next_af && (secondary_lobe[0] < 0 || current_af > secondary_lobe[0]) && main_lobe[0] - current_af > 0.5)
        {
            if (theta == 0 && phi == 0 && flag_theta_zero == false) {
                secondary_lobe[0] = current_af;
                secondary_lobe[1] = theta;
                secondary_lobe[2] = phi;
                flag_theta_zero = 1;
            }
            else if (theta == M_PI && phi == 0 && flag_theta_pi == false) {
                secondary_lobe[0] = current_af;
                secondary_lobe[1] = theta;
                secondary_lobe[2] = phi;
                flag_theta_pi = 1;
            }
            else if ((theta == 0 && flag_theta_zero == true) || (theta == M_PI && flag_theta_pi == true)) {
                previous_af=current_af;
                current_af=next_af;
                continue;
            }
            else
            {
                previous_af_phi=array_factor(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y, theta, phi - delta_phi);
                next_af_phi=array_factor(active_elements_array, i_amplitude_array, d_x, d_y, b_x, b_y, n_x, n_y, theta, phi + delta_phi);
                if (current_af > previous_af_phi && current_af > next_af_phi) {
                    secondary_lobe[0] = current_af;
                    secondary_lobe[1] = theta;
                    secondary_lobe[2] = phi;
                }
            }
            
        }
        previous_af=current_af;
        current_af=next_af;


        }

    }
    if (secondary_lobe[0]<0)
    {
        secondary_lobe[0] = main_lobe[0];
        secondary_lobe[1] = main_lobe[1];
        secondary_lobe[2] = main_lobe[2];
        
    }

    double sll=20*log10(secondary_lobe[0] / main_lobe[0]);
    return sll;
    
}


// Test code

// int main() {
//     int nx = 16;
//     int ny = 16;
//     double bx = 0;
//     double by = 0;
//     double dx = 1.0 / 2;
//     double dy = 1.0 / 2;
//     double theta0 = 0;
//     double phi0 = 0;
//     int active_el[256];
//     int active_ampl[256];
//     for (int i = 0; i < 256; i++) {
//         active_el[i] = 1;
//         active_ampl[i] = 1;
//     }
//     // clock_t start = clock();
//     // double  sll = array_factor_numba(active_el, active_ampl, dx, dy, bx, by, nx, ny, 0, 0);
//     // printf("SLL = %.2f \n", sll);
//     // printf("Time %.6f sec\n", (double)(clock() - start) / CLOCKS_PER_SEC);
    
//     clock_t start = clock();
//     double  dir = directivity_planar_array_antenna(active_el, active_ampl, dx, dy, bx, by, nx, ny, 0, 0);
//     printf("Dir = %.2f \n", dir);
//     printf("Time %.6f sec\n", (double)(clock() - start) / CLOCKS_PER_SEC);

//     start = clock();
//     double  sll = sll_planar_array_antenna(active_el, active_ampl, dx, dy, bx, by, nx, ny);
//     printf("SLL = %.2f \n", sll);
//     printf("Time %.6f sec\n", (double)(clock() - start) / CLOCKS_PER_SEC);

//     double a=sin(M_PI/2);
//     printf("%.2f",a);
//     return 0;
// }