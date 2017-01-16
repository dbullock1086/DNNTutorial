#!/usr/bin/env python

# hPyROOT jobs will all start with importing the necessary modules and objects
import os # access to environment variables
from eventloop.Driver import * # this collects all job info
from eventloop.EventStore import * # this will evaluate expressions involving data
from pyobjects.VarDef import * # define raw dataset labels to read
from pyobjects.LorentzDef import * # create a four-vector directly from data

# To facilitate calculating new observables from raw four-vectors, we employ the
# use of two algorithms:
from elalgs.InvMass import * # calculate invariant mass
from elalgs.DeltaPhi import * # calculate azimuthal angle

# And we want to be able to write these calculated variables as histograms in a
# ROOT file:
from elalgs.HistFill import * # histogram filling

# Because there are two datasets in the HDF5 file and we want to represent both,
# we use a for loop.
for study in ['Zll', 'Rndm']:
    print '*'*64
    print 'PERFORMING STUDY:', study
    print '*'*64

    if study == 'Zll': tname = 'Zto2LOS'
    else: tname = 'Rndm2LOS'

    # Now we initialize the Driver and set the I/O:
    driver = Driver ('raw' + study)
    driver.Input (os.getenv('SampleDirZll') + '/Zll.h5', tname)
    driver.Output (os.getenv('HistDir'))

    # hPyROOT supports creating expressions from an arbitrary construction of
    # labels (columns) in the dataset. By defining a set of pyobjects and passing
    # them to this EventStore, they can be evaluated for each event and are made
    # available for subsequent algorithms to use them.
    estore = EventStore ('estore')

    # One such pyobject just makes a direct copy of one value from the dataset.
    # Notice that you can define a new name to associate with the label.
    LPphi = VarDef ('LPphi', 'LP_phi')
    estore.AddVar (LPphi)

    LMphi = VarDef ('LMphi', 'LM_phi')
    estore.AddVar (LMphi)

    # Alternatively, you can define four-vectors by component, and these
    # components are updated per event.
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
    
    # We use an algorithm to calculate the invariant mass and dphi from 2 four-
    # vectors.
    Mll = InvMass ('Mll', 'LP', 'LM')
    driver.Alg (Mll)

    DPhi = DeltaPhi ('DPhi', 'LP', 'LM')
    driver.Alg (DPhi)

    # Then use these new variables to fill histograms. We have 2D histograms of
    # dphi for the two objects:
    h_LPphi_LMphi = HistFill ('h_LPphi_LMphi',
                              'LPphi', 20, -3.1416, 3.1416,
                              'LMphi', 20, -3.1416, 3.1416)
    driver.Alg (h_LPphi_LMphi)

    # And 1D histograms of invariant mass and dphi.
    h_InvMass = HistFill ('h_Mll',
                          'Mll', 20, 0, 200)
    driver.Alg (h_InvMass)

    h_DPhi = HistFill ('h_DPhi',
                       'DPhi', 20, 0, 3.1416)
    driver.Alg (h_DPhi)

    # Once the job is properly configured, submit it.
    driver.Submit ()
    pass
