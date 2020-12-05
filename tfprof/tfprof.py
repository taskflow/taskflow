#!/usr/bin/env python3

# program: tfprof

import logging as logger
import time
import sys
import json
import argparse
import os
import subprocess
import requests

# run_tfprof (default)
# generate profiler data in taskflow profiler format
def run_tfprof(args):

  args.output = os.path.abspath(args.output);
  
  logger.info("profiling program \"" + ' '.join(args.program) + "\"")
  
  ## open the output file
  with open(args.output, "w") as ofs:
  
    ofs.write('[');
  
    os.environ["TF_ENABLE_PROFILER"] = args.output;
    
    ## launch the program
    prob = time.perf_counter();
    subprocess.call(args.program);
    proe = time.perf_counter();
    logger.info(f"finished with {(proe - prob)*1000:0.2f} milliseconds");
    logger.info(f"saved result to {args.output:s}");
  
  if(args.port == None): 
    return;

  logger.info(f"sending the result to localhost:{args.port:d}");

# run_chrome (TODO)
# generate the profiler data in chrome tracing format

# main function
def main():

  # configure logger
  logger.basicConfig(
    #format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    format='%(asctime)s %(levelname)s: %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S',
    level=logger.DEBUG
  )
  
  # parse the input arguments
  parser = argparse.ArgumentParser();
  
  parser.add_argument(
    '-o', '--output',
    type=str,
    help='file to save the result (default: output.tfp)',
    default="output.tfp"
  )

  parser.add_argument(
    '-p', '--port',
    type=int,
    help='port number of the profiler server (default: None)',
    default=None
  )
  
  parser.add_argument(
    'program', 
    nargs=argparse.REMAINDER,
    help='program to profile (e.g., path/to/binary args)'
  )
  
  args = parser.parse_args();
  
  if(len(args.program) == 0) :
    logger.error("no program specified");
    sys.exit(1);

  run_tfprof(args);


# main entry
if __name__ == "__main__":
  main();


