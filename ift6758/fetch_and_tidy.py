#!/usr/bin/env python3

import subprocess
import os
import features.data_formatting
from re import match
from time import sleep

def subprocess_popen_cmanager(args: list, timeout: int):
    """
    Passes `args` to subprocess.Popen and defines a context manager around it

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
    """
    Boilerplate code to ensure validity of passed args to NHLDataFetcher

    Returns:
        None if no arguments,
        NHLDataFetcher helper if REGEX pattern `-?-?he?.?` is matched against arguments,
        Arguments as if, if they are valid
    """
    passed_args = os.sys.argv[1:] if len(os.sys.argv) > 1 else None
    if passed_args is None:
        return None
    if any([ match('-?-?he?.?', arg) for arg in passed_args ]):
        subprocess.Popen(['python', 'data/main.py', '--help'])
        print('Exiting..')
        os.sys.exit()
    p = subprocess_popen_cmanager(['python','data/main.py', '--parse-args', *passed_args], timeout=30)
    args = p.args
    if '--parse-args' in p.args:
        args.remove('--parse-args')

    return args

def main():
    passed_args = check_for_cli_args()
    print("Attempting to download games")
    subprocess_popen_cmanager(passed_args or ['python', 'data/main.py', '-y', '2016-2023', '-t', '2,3'], timeout=1800)
    print("Filtering json, formatting to pandas DataFrame and saving to csv")
    features.data_formatting.process_and_save_json_file(*features.data_formatting.gather_and_check_paths())

if __name__ == '__main__':
    main()
