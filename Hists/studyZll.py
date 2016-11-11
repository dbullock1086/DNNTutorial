#!/usr/bin/env python

import os

# load all scripts necessary for an Event Loop job
from eventloop.Driver     import *
from eventloop.EventStore import *
from pyobjects.VarDef     import *
from pyobjects.LorentzDef import *
from elalgs.InvMass       import *
from elalgs.HistFill      import *

# set up the event driver
# including the input and output
driver = Driver ('makehists')
driver.Input (os.getenv('TrainDir') + '/ZllModel_0/Result.h5', 'tree')
driver.Output (os.getenv('HistDir') + '/studyZll.root')

#### start feeding algorithms into the driver

# define an eventstore to hold the input data
estore = EventStore ('estore')

# prediction term
predict = VarDef ('predict_0', 'prediction')
estore.AddVar (predict)

# four vectors:
LP = LorentzDef ('LP',
                 ['LP_pT', 'LP_eta', 'LP_phi', 'LP_E'],
                 mode='ptetaphie')
estore.AddVar (LP)

LM = LorentzDef ('LM',
                 ['LM_pT', 'LM_eta', 'LM_phi', 'LM_E'],
                 mode='ptetaphie')
estore.AddVar (LM)

driver.Alg (estore)

# calculate some new observables
Mll = InvMass ('Mll', 'LP', 'LM')
driver.Alg (Mll)

# fill a histogram with values
h = HistFill ('h_feature',
              'Mll', 20, 0, 200,
              'predicttion', 20, 0, 1)
driver.Alg (h)

# run the job
driver.Submit ()
