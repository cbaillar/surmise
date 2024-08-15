
import numpy as np
import os
import subprocess


def fit(x, theta, f):
    
    nx = x.shape[0]
    write_observable_info(x, exp_error = np.full(nx, 0.0))
    write_obs_txt(nx, f)
    write_mod_par(theta)

    subprocess.run(['./smoothy_tune'], check=True)  

    return

def write_observable_info(x, exp_error):
    path = 'smooth_data/Info'
    filename = 'observable_info.txt'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), 'w') as out_file:
        for i in range(len(x)):
            out_file.write(f"{x[i]} {exp_error[i]}\n")

def write_mod_par(theta):
    filename = 'TrainingThetas.txt'
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w') as out_file:
        for row in theta:
            row_str = ' '.join(map(str, row))
            out_file.write(f"{row_str}\n")

def write_obs_txt(nx, f):
    filename = 'TrainingObs.txt'
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w') as out_file:
        for i in range(len(f[0:])):
            for j in range(nx):
                out_file.write(f"{f[i][j]} ")
            out_file.write("\n")


def predict(x, params):
    y_pred = []
    error = []
    results = []
    

    for xi in x:
        for param_row in params:
            args = [str(xi)] + list(map(str, param_row))
            command = ['./smoothy_surmise_calcobs'] + args
                
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            result_lines = result.stdout.strip().split()
            y_pred.append(list(map(float, result_lines[:len(x)])))
            error.append(list(map(float, result_lines[len(x):])))

    return y_pred, error
   
