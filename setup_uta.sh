#!/bin/bash

# establish environment (dependent and user)
export DepWork=/home/dbullock
export UserWork=/home/dbullock/DNNTutorial

# establish the environment
export MLDir=$HOME/.virtualenvs/keras
export HPYROOTDIR=$DepWork/hPyROOT
export PATH=$MLDir/bin:$PATH

export SampleDirZll=$DepWork/samples
export SampleDirSUSY=/scratch/data-backup/afarbin/crogan/h5

export TrainDir=$Tutorial/DLKit/TrainedModel
export HistDir=$Tutorial/hists

source /setups/setup_virtualenv.sh
source /setups/setup_cuda-8.0.sh
source /setups/setup_root.sh
#source /opt/root-5.34/bin/thisroot.sh
source activate keras

export PATH=$HPYROOTDIR/bin:$PATH
export PYTHONPATH=$HPYROOTDIR/python:$PYTHONPATH
