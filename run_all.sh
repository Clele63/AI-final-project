#!/bin/sh

set -e

# sbatch run_step1.sbatch
# sbatch --dependency=afterok: run_step2.sbatch