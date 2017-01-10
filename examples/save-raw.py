#!/usr/bin/env python

# Start by importing these modules:
import os
from matplotlib import pyplot as plt
from rootpy.plotting import root2matplotlib as rplt
from rootpy.io import root_open

# Remember that we have two classifications to view:
hists = {}
for a in ['Zll', 'Rndm']:
    # Open the ROOT file:
    f = root_open(os.getenv('HistDir') + '/study%s.root' % a)

    # We have three histograms from each classification:
    hists[a] = {}
    for b in ['LPphi_LMphi',
              'Mll', 'DPhi']:
        # Load the histograms into memory and rename to avoid conflicts.
        hists[a][b] = f.Get('h_%s' % b)
        hists[a][b].SetDirectory(0)
        hists[a][b].SetName('%s_%s' % (a,b))

        # Give the histograms a little style:
        if b in ['Mll', 'DPhi']:
            if a == 'Zll':
                title = r'$Z \rightarrow \ell\ell$'
                color = 2
                pass
            else:
                title = r'$Rndm \rightarrow \ell\ell$'
                color = 4
                pass
            hists[a][b].SetTitle(title)
            hists[a][b].SetFillColor(color)
            hists[a][b].SetLineWidth(2)
            hists[a][b].SetLineColor(color)
            hists[a][b].SetMarkerColor(color)
            pass
        pass
    # And close the files:
    f.Close()
    pass

# Show the 2D histograms:
for a in ['Zll', 'Rndm']:
    print a, 'LPphi_LMphi'
    fig, ax = plt.subplots()
    ax.set_title(r'Comparison of $\ell_\phi$')
    ax.set_xlabel(r'$\ell_{\phi}^{+}$', fontsize=16)
    ax.set_ylabel(r'$\ell_{\phi}^{-}$', fontsize=16)
    hdraw = rplt.imshow(hists[a]['LPphi_LMphi'])
    cb = fig.colorbar(hdraw)
    plt.show()
    pass

# Show the 1D histograms:
for b in ['Mll', 'DPhi']:
    print b
    fig, ax = plt.subplots()
    ax.set_title('Comparison of %s' % b)
    if b == 'Mll': ax.set_xlabel(r'$M_{\ell\ell}$', fontsize=16)
    else: ax.set_xlabel(r'$\Delta\phi$', fontsize=16)
    ax.set_ylabel('Entries')
    for a in ['Zll', 'Rndm']: rplt.hist(hists[a][b])
    ax.legend(loc=2)
    if b == 'Mll':
        plt.plot([91.1876,91.1876], ax.get_ylim(), 'k')
        ax.text(95, 325000, r'$M_{\ell\ell}=m_{Z}$', fontsize=16)
        pass
    plt.show()
    pass
