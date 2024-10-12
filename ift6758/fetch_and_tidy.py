#!/usr/bin/env python3
""" import data.main import features.data_formatting

#Fetch all uncached games
data.main -y 2022 -t 2 -g 1-130

#Tidy JSONs, parse as Pandas dataframe and save to csv
features.data_formatting

"""
import subprocess
import os
import features.data_formatting
from time import sleep

def subprocess_popen_cmanager(args: list, timeout: int):
    """
    Passed `args` to subprocess.Popen and defines a context manager around it

    Parameters:
        args: list, list of command arguments
        timeout: int, timeout in seconds

    Return:
        p: subprocess.Popen, finished process of args
    """
    print(f'Opening process for command {args}')
    print(f'Setting timeout to {timeout} seconds')
    try:
      p = subprocess.Popen(args)
      p.communicate(timeout=timeout)
    except KeyboardInterrupt:
      print('CTRL-C catched, terminating program')
      p.terminate()
      sleep(5)
      #If SIGTERM does not do the trick, force raise TimeoutExpired
      if not p.poll():
          raise p.TimeoutExpired()
    except subprocess.TimeoutExpired:
        print('Timeout for program execution reached, killing program')
        p.kill()
    finally:
        return p

def check_for_cli_args():
    passed_args = os.sys.argv[1:] if len(os.sys.argv) > 1 else None
    if passed_args is None:
        return None
    if 'help' in passed_args:
        subprocess.Popen(['python', 'data/main.py', '--help'])
        return
    """
    try:
        p = subprocess.Popen(['python','data/main.py', '--parse-args', *passed_args],
                             capture_output = True)
        #Set timeout to 30*60 seconds
        p.communicate(timeout=(30*60))
    #Capture CTRL-C
    except KeyboardInterrupt:
      try:
        print('CTRL-C catched, terminating program')
        p.terminate()
        sleep(5)
        #If SIGTERM does not do the trick, force raise TimeoutExpired
        if not p.poll():
            raise p.TimeoutExpired()
      #If original timeout expire or if soft termination did not end the process
 S     except TimeoutExpired:
        print('Timeout for soft termination reached, killing program')
        p.kill()

    except TimeoutExpired:
        print('Timeout for program execution reached, killing program')
        p.kill()
    """
    p =subprocess_popen_cmanager(['python','data/main.py', '--parse-args', *passed_args], timeout=30)
    args = p.args
    if '--parse-args' in p.args:
        args.remove('--parse-args')

    return args

def main():
    print("Attempting to download games")
    passed_args = check_for_cli_args()
    subprocess_popen_cmanager(passed_args or ['python', 'data/main.py', '-y', '2016-2023', '-t', '2,3'], timeout=1800)
    #subprocess.Popen(passed_args or ['python', 'data/main.py', '-y', '2016-2023', '-t', '2,3'])
    print("Filtering json, formatting to pandas DataFrame and saving to csv")
    features.data_formatting.process_and_save_json_file(*features.data_formatting.gather_and_check_paths())

if __name__ == '__main__':
    main()
