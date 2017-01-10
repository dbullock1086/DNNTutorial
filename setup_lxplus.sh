#!/bin/bash

# establish environment (dependent and user)
export DepWork=/afs/cern.ch/work/d/dbullock/public
export UserWork=/afs/cern.ch/work/d/dbullock/public/DNNTutorial

# establish the environment
export MLDir=$DepWork/MLBase/miniconda3/envs/testenv
export HPYROOTDIR=$DepWork/hPyROOT
export PATH=$MLDir/bin:$HPYROOTDIR/bin:$PATH
export PYTHONPATH=$HPYROOTDIR/python:$PYTHONPATH

export SampleDirZll=$DepWork/samples/dnn-tutorial
export SampleDirSUSY=/afs/cern.ch/work/a/afarbin/public/RestFrames-llbbMET

export TrainDir=$UserWork/DLKit/TrainedModels
export HistDir=$UserWork/hists

source activate testenv
cd $MLDir
source bin/thisroot.sh
cd $UserWork
