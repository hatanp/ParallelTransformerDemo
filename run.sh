#!/bin/bash
##qsub -A datascience -q lustre_scaling -l select=1 -l walltime=03:00:00,filesystems=home -I



export CPU_BIND="list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96"
NNODES=`wc -l < $PBS_NODEFILE`
PPN=12
cd /flare/Aurora_deployment/vhat/gb25_cli/ParallelTransformerDemo
mpiexec --verbose --envall -n 1 -ppn 1 --cpu-bind $CPU_BIND python3 test.py