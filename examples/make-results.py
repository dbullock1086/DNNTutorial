#!/usr/bin/env python

import os # access to environment variables
from eventloop.Driver import * # this collects all job info
from eventloop.EventStore import * # this will evaluate expressions involving data
from pyobjects.VarDef import * # define raw dataset labels to read
from pyobjects.LorentzDef import * # create a four-vector directly from data

from elalgs.InvMass import * # calculate invariant mass
from elalgs.DeltaPhi import * # calculate azimuthal angle

from elalgs.HistFill import * # histogram filling

# Now we initialize the Driver and set the I/O:
driver = Driver ('studyZll')
driver.Input (os.getenv('TrainDir') + '/ZllModel_100k/Result.h5', 'tree')
driver.Output (os.getenv('HistDir'))

# hPyROOT supports creating expressions from an arbitrary construction of labels
# (columns) in the dataset. By defining a set of pyobjects and passing them to
# this EventStore, they can be evaluated for each event and are made available
# for subsequent algorithms to use them.
estore = EventStore ('estore')

# One such pyobject just makes a direct copy of one value from the dataset. We
# just need to know the prediction outcome. Notice the new nomenclature
# associated with the label.
predict = VarDef ('predict', 'predict_0')
estore.AddVar (predict)

truth = VarDef ('truth', 'true_0')
estore.AddVar (truth)

# Alternatively, you can define four-vectors by component, and these components
# are updated per event.
LP = LorentzDef ('LP',
                 ['LP_pT', 'LP_eta', 'LP_phi', 'LP_E'],
                 mode='ptetaphie')
estore.AddVar (LP)

LM = LorentzDef ('LM',
                 ['LM_pT', 'LM_eta', 'LM_phi', 'LM_E'],
                 mode='ptetaphie')
estore.AddVar (LM)

# Once you have all the objects you need, add the EventStore to the Driver.
driver.Alg (estore)

# We use an algorithm to calculate the invariant mass and dphi from 2
# four-vectors.
Mll = InvMass ('Mll', 'LP', 'LM')
driver.Alg (Mll)

DPhi = DeltaPhi ('DPhi', 'LP', 'LM')
driver.Alg (DPhi)

# Create 2D histograms
h_InvMass = HistFill ('h_Mll',
                      'Mll', 20, 0, 200,
                      'predict', 21, 0, 1.05)
driver.Alg (h_InvMass)

h_DPhi = HistFill ('h_DPhi',
                   'DPhi', 20, 0, 3.1416,
                   'predict', 21, 0, 1.05)
driver.Alg (h_DPhi)

h_perf = HistFill ('h_perf',
                   'predict', 21, 0, 1.05,
                   'truth', 2, 0, 2)
driver.Alg (h_perf)

# And submit once you're done
driver.Submit ()
