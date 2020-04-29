#!/bin/bash -l

# Batch script to run a serial array job under SGE.

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:20:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=2G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)

# Set up the job array.  In this instance we have requested 10000 tasks
# numbered 1 to 10000.
#$ -t 1-4

# Set the name of the job.
#$ -N MyArrayJob

# Set the working directory to somewhere in your scratch space. 
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucapsjg/workdir

# Run the application.

echo "$JOB_NAME ($SGE_TASK_ID - 1)"

