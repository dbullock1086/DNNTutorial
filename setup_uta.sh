#!/bin/bash

# establish environment (dependent and user)
export DepWork=/home/dbullock
export UserWork=/home/dbullock/DNNTutorial

# establish the environment
export MLDir=$DepWork/.virtualenvs/keras
export HPYROOTDIR=$DepWork/hPyROOT

export SampleDir=$DepWork/samples
#export SampleDir=/scratch/data-backup/afarbin/crogan/h5/

export TrainDir=$Tutorial/DLKit/TrainedModel
export HistDir=$Tutorial/Hists

source /setups/setup_virtualenv.sh
source /setups/setup_cuda-8.0.sh
source /setups/setup_root.sh
source activate keras

export PATH=$HPYROOTDIR/bin:$PATH
export PYTHONPATH=$HPYROOTDIR/python:$PYTHONPATH
