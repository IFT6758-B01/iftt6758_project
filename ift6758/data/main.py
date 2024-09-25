#!/usr/bin/env python3

from data_acquisition import NHLDataFetcher
from typing import Union
import argparse
import pathlib
import os
import re
import sys

def init_parser():
  #Instanciate parser
  parser = argparse.ArgumentParser(
    prog='NHLDataFetcher',
    description="Query and store NHL API requests"
  )

  parser.add_argument(
    '-y', '--year',
    type=str,
    default='all',
    help="Year of games wanted"
  )

  parser.add_argument(
    '-t', '--type',
    type=str,
    default="all",
    choices=['2', '3'],
    help="Type of games wanted ('regular': 2, 'playoffs': 3)"
  )

  parser.add_argument(
    '-g', '--games',
    type=str,
    default='all',
    help='Games ID wanted as interval (ex : 0-100)'
  )

  parser.add_argument(
    '-o', '--output',
    type=pathlib.Path,
    default=f'{pathlib.Path.cwd()}',
    help='Directory in which output should go to. Defaults to current directory'
  )

  subparser = parser.add_mutually_exclusive_group(required=False)
  subparser.add_argument(
    '-p', '--pretty',
    dest='pretty',
    action='store_true',
    help='Pretty-print JSON in output file'
  )
  subparser.add_argument(
    '--no-pretty',
    dest='pretty',
    action='store_false'
  )
  parser.set_defaults(pretty=True)
  return parser


def reg_match_arg(arg: str) -> list:
  try:
    #Parse NUM,...,NUM	Series
    if re.match(
     r"""
      \d+,(\d+,?){1,} #2+ patterns of digits with optional ',' after 1 occurrence
     """,
     arg,
     re.VERBOSE) :
      match_list = arg.split(',')
      for i in range(len(match_list) - 1 ):
        assert int(match_list[i+1]) > int(match_list[i]), f'{match_list[i+1]} !> {match_list[i]}'
      return match_list
    #Parse NUM-NUM			Interval
    elif re.match(
     r"""
      ^\d+    #4digits for gid_min
      \-        #dash as interval
      \d+$    #4 digits for gid_max
     """,
     arg,
     re.VERBOSE) :
      val_min, val_max = arg.split('-')
      assert int(val_max) > int(val_min), f'{val_max} !> {val_min}'
      return [val for val in range(int(val_min), int(val_max)+1)]
    #Parse NUM					Single
    elif re.match(r'^\d+$', arg):
      return [arg]
  except AssertionError as e:
    print(e)


def verify_args_parser(parser, args : argparse.Namespace = None):
  if args == None:
    args = parser.parse_args()
  try:
    assert args.year, 'Missing argument year'
    assert args.type, 'Missing argument type'
    assert args.output, 'Missing argument output'
    q_year = range(2015,2024) if args.year == parser.get_default('year') else reg_match_arg(args.year)
    #if args.year != parser.get_default('year'):
    #  q_year = reg_match_arg(args.year)
    q_type = [ f'{SOME:02d}' for SOME in (2,3) ] if args.type == parser.get_default('type') else reg_match_arg(args.type)
    #if args.type != parser.get_default('type'):
    #  q_type = reg_match_arg(args.type)
    if args.output != parser.get_default('output'):
      assert pathlib.Path.is_dir(args.output), f'{args.output} is not a directory'
    q_dir = args.output
    return q_year, q_type, q_dir
  except AssertionError as err:
    print(err)
  except ValueError as err:
    print(err)

def main():
  #Instanciate parser
  parser = init_parser()
  #Parse arguments
  sys_args = sys.argv[1:]
  print(sys_args)
  input('Continue?')
  args = parser.parse_args(sys_args)
  #Check if valid arguments
  q_year, q_type, q_dir = verify_args_parser(parser, args)
  #Instanciate fetcher
  base_url = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
  fetcher = NHLDataFetcher(base_url, save_dir = q_dir or None)
  #Execute fetcher
  for season in q_year:
    fetcher.get_season_data(season)


if __name__ == '__main__':
  main()
