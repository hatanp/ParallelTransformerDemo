#!/bin/bash

cat $PBS_NODEFILE > hostfile

source /flare/Aurora_deployment/vhat/gb25_cli/arrgen/cp_mv.sh

module unload oneapi/eng-compiler/2024.07.30.002
module use /opt/aurora/24.180.3/spack/unified/0.8.0/install/modulefiles/oneapi/2024.07.30.002
module use /soft/preview/pe/24.347.0-RC2/modulefiles
module add oneapi/release


export PALS_PING_PERIOD=240
export PALS_RPC_TIMEOUT=240

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export CCL_PROCESS_LAUNCHER=pmix    
export PALS_PMI=pmix                
export CCL_ATL_TRANSPORT=mpi       
export CCL_OP_SYNC=1                

export FI_PROVIDER=cxi

export I_MPI_OFI_LIBRARY="/opt/cray/libfabric/1.20.1/lib64/libfabric.so.1" 

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

source /flare/Aurora_deployment/vhat/gb25_cli/arrgen/ccl_env.sh

source /home/vhat/miniconda3/etc/profile.d/conda.sh
conda activate /tmp/sc25_gb_cli_conda_env

ulimit -n 524288