#!/bin/bash -x
srun hostname -s > /tmp//hosts.$SLURM_JOB_ID
if [ "x$SLURM_NPROCS" = "x" ]
then
if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
then
 SLURM_NTASKS_PER_NODE=1
fi
SLURM_NPROCS=`expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE`
fi
python -hostfile /tmp/hosts.$SLURM_JOB_ID -np $SLURM_NPROCS ./a.out
rm /tmp/hosts.$SLURM_JOB_ID