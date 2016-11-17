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
driver.Input (os.getenv('TrainDir') + '/SUSY_Physics/Result.h5', 'tree')
driver.Output (os.getenv('HistDir') + '/studySUSY.root')

#### start feeding algorithms into the driver

# define an eventstore to hold the input data
estore = EventStore ('estore')

# prediction term
predict = VarDef ('prediction', 'predict_0')
estore.AddVar (predict)

#['mP', 'mC', 'mX',
# 'METx', 'METy',
# 'L1_pT', 'L1_eta', 'l1_phi', 'L1_M',
# 'L2_pT', 'L2_eta', 'L2_phi', 'L2_M',
# 'B1_pT', 'B1_eta', 'B1_phi', 'B1_M',
# 'B2_pT', 'B2_eta', 'B2_phi', 'B2_M',

driver.Alg (estore)

# fill a histogram with values
h = HistFill ('h_feature',
              'Mll', 20, 0, 200,
              'prediction', 20, 0, 1)
driver.Alg (h)

# run the job
driver.Submit ()
