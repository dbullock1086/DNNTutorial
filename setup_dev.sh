#!/bin/bash

setupCode hPyROOT
setupCode RestFrames

setupCode WorkDir $HOME/Work/DNNTutorial
setupCode HistDir $WorkDir/Hists

setupCode SampleDir $HOME/Samples/dnn-tutorial

export ResultDir=$HOME/Samples/trained
