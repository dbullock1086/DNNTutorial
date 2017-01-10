#!/bin/bash

setupCode hPyROOT
setupCode RestFrames

setupCode WorkDir $HOME/Work/DNNTutorial
setupCode HistDir $WorkDir/hists

export SampleDirZll=$HOME/Samples/dnn-tutorial
export SampleDirSUSY=$HOME/Samples/dnn-tutorial

export TrainDir=$HOME/Samples/trained
