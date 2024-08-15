import numpy as np
import os
import subprocess

def fit(x, theta, f, priors):
    nx = x.shape[0]
    write_observable_info(x, exp_error = np.full(nx, 0.0))
    update_parameters_in_file(priors)

    #Store 'smooth_data/Info' parameters
    write_obs_txt(nx, f)
    write_mod_par(theta)

    #trains the emulator
    subprocess.run(['./smoothy_tune'], check=True)  
    return

def write_observable_info(x, exp_error):
    #Saves observable(the x values) in the first column and their errors in second column
    path = 'smooth_data/Info'
    filename = 'observable_info.txt'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), 'w') as out_file:
        for i in range(len(x)):
            out_file.write(f"{x[i]} {exp_error[i]}\n")

def update_parameters_in_file(priors):
    path = 'smooth_data/Info'
    filename = 'modelpar_info.txt'
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, filename)

    with open(file_path, 'w') as out_file:
        # Write the header line
        out_file.write("# par_name dist_type xmin  xmax\n")
        
        for param_name, values in priors.items():
            # Assuming 'values' contains 'xmin' and 'xmax'
            xmin = str(values['xmin'])
            xmax = str(values['xmax'])
            # Write the parameter information to the file
            out_file.write(f"{param_name} uniform {xmin} {xmax}\n")

def write_mod_par(theta):
    #makes file and saves training parameters. row for each parameter set
    filename = 'TrainingThetas.txt'
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w') as out_file:
        for row in theta:
            row_str = ' '.join(map(str, row))
            out_file.write(f"{row_str}\n")

def write_obs_txt(nx, f):
    #makes file and saves predictions for training parameters. column for each x value and row for each parameter set.
    filename = 'TrainingObs.txt'
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w') as out_file:
        for i in range(len(f[0:])):
            for j in range(nx):
                out_file.write(f"{f[i][j]} ")
            out_file.write("\n")

def predict(nx, params):
    y_pred = []
    error = []
    results = []
    
    for param_row in params:

        args = list(map(str, param_row))
        command = ['./smoothy_surmise_calcobs'] + args
        
        #Imput: single parameter set 
        #Outputs: predictions for each x value
        #calls for trained emulator's predictions.
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Split the output into lines
        result_lines = result.stdout.strip().split()
        
        # Extract y_pred and error values
        y_pred.append(list(map(float, result_lines[:nx])))
        error.append(list(map(float, result_lines[nx:])))

    return y_pred, error
