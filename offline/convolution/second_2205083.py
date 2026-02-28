import numpy as np
import matplotlib.pyplot as plt
import first_2205083

file_path = 'input_signal.txt'

n_start = None
n_end = None
input_signal = []

try:

    with open(file_path, 'r') as file:

        line_number = 0
        for line in file:

            elements = line.strip().split()

            if line_number == 0:    
                n_start = int(elements[0])
                n_end = int(elements[1])
                line_number += 1
            else:
                for element in elements:
                    input_signal.append(int(element))

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")


INF = max(abs(n_start), abs(n_end))
INF = INF + 10
signal = first_2205083.Signal(INF)


i = 0
for k in range(n_start, n_end+1):

    signal.set_value_at_time(k,input_signal[i])
    i+=1

signal.plot("input_signal")

impulse_response = first_2205083.Signal(INF)

for i in range(-2, 3):
    impulse_response.set_value_at_time(i, .2)

impulse_response.plot("impulse_response_LTI_system")

LTI_system = first_2205083.LTI_System(impulse_response)

output_signal = LTI_system.output(signal)

output_signal.plot("smoothed_output_signal")










        
    