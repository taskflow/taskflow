#!/usr/bin/env python3

# program: tfprof

import logging as logger
import time
import sys
import json
import argparse
import os
import tempfile
import subprocess

# run_tfprof (default)
# generate profiler data in taskflow profiler format
def run_tfprof(program, output):
  
  tfpb = time.perf_counter(); 
  
  logger.info("profiling program \"" + ' '.join(program) +'\"')
  
  with open(output, "w") as ofs:
  
    ofs.write('[');
  
    with tempfile.TemporaryDirectory() as dirname:
    
      prefix = os.path.join(dirname, 'executor-')
    
      os.environ["TF_ENABLE_PROFILER"] = prefix;
      
      prob = time.perf_counter();
      subprocess.call(program);
      proe = time.perf_counter();
      logger.info(f"program finished in {(proe - prob)*1000:0.2f} milliseconds")
  
      executor_id = 0;
      logger.info("collecting profiled data ...")
      for fnum, fname in enumerate(os.listdir(dirname)):
        if fname.endswith(".tfp") : 
          ifile = os.path.join(dirname, fname);
          logger.info(f"processing {ifile:s} ...")
          with open(ifile, "r") as ifs:
            data = json.load(ifs);
            data['executor'] = executor_id;
            executor_id = executor_id + 1;
            if fnum != 0 :
              ofs.write(',');
            #ofs.write(ifs.read());
            json.dump(data, ofs)
  
    ofs.write("]\n");
  
  logger.info(f"saved result to {output:s}");
  
  tfpe = time.perf_counter(); 
      
  logger.info(f"tfprof finished in {(tfpe - tfpb)*1000:0.2f} milliseconds")


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
    help='file to save the result (default output.tfp)',
    default="output.tfp"
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
  
  run_tfprof(args.program, args.output);


# main entry
if __name__ == "__main__":
  main();









