import logging
import cantools
import subprocess
import sys
import os
def convert_can_logs(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        first_line = True
        for line in f_in:
            if first_line:
                first_line=False
                continue
            parts = line.strip().split(',')
            timestamp_sec = float(parts[0]) / 1000  # Convert milliseconds to seconds
            msg_id = parts[1]
            msg_len = int(parts[2])
            data = parts[3]

            # Convert message ID and length to hexadecimal format
            msg_id_hex = f'{int(msg_id, 16):X}'
            msg_len_hex = f'{msg_len:X}'

            # Modify the format of the data section
            data_formatted = data

            # Write the converted log entry to the output file
            f_out.write(f'({timestamp_sec:.6f}) can0 {msg_id_hex.zfill(3)}#{data_formatted}\n')
    return output_file

def cantools_plot_csv(input_file,dbc,plotargs):
    if plotargs is None:
        subprocess.run['cantools','plot']
    elif plotargs is not None:
        cat_command = ['cmd.exe','cat', 'raw_data.CSV.log']

# Command for the other process (e.g., 'grep')
        other_process_command = ['cantools', 'plot','.\\dbc-files\\ksu_ev_can.dbc']
        file = convert_can_logs(input_file=input_file,output_file=(input_file+".log"))
        # subprocess.run(['cmd.exe','cat',str(file),'|','python', "-m", "cantools", "plot",dbc, plotargs])
        with subprocess.Popen(cat_command, stdout=subprocess.PIPE) as cat_proc:
            with subprocess.Popen(other_process_command, stdin=cat_proc.stdout) as other_proc:
                other_proc.communicate()
if __name__ == "__main__":
    file = sys.argv[1]
    argsz = sys.argv[2]
    dbc=sys.argv[3]
    print(sys.argv)
    cantools_plot_csv(file,dbc,argsz)