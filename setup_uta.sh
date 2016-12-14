#!/bin/bash

# establish environment (dependent and user)
export DepWork=/home/dbullock
export UserWork=/home/dbullock/DNNTutorial

# establish the environment
export MLDir=$DepWork/.virtualenvs/keras
export HPYROOTDIR=$DepWork/hPyROOT
export PATH=$MLDir/bin:$PATH

export SampleDir=$DepWork/samples
#export SampleDir=/scratch/data-backup/afarbin/crogan/h5
#export SampleDir=/scratch/pjackson/SignalBGCompressedSkim

export TrainDir=$Tutorial/DLKit/TrainedModel
export HistDir=$Tutorial/Hists

source /setups/setup_virtualenv.sh
source /setups/setup_cuda-8.0.sh
source /setups/setup_root.sh
#source /opt/root-5.34/bin/thisroot.sh
source activate keras

export PATH=$HPYROOTDIR/bin:$PATH
export PYTHONPATH=$HPYROOTDIR/python:$PYTHONPATH
