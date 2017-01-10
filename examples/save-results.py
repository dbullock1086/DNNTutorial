#!/usr/bin/env python

import os
from numpy import nan
from matplotlib import pyplot as plt
from rootpy.plotting import root2matplotlib as rplt
from rootpy.io import root_open

# Open the file
f = root_open(os.getenv('HistDir') + '/studyZll.root')

# make plots
for p in ['Mll', 'DPhi', 'pref']:
    h = f.Get('h_%s' % p)
    # By default, empty bins will be drawn as if their content is zero. This can
    # be problematic since it can imply that these conditions are equivalent.
    # However, when the DNN decides that a low number of events from one bin are
    # predicted to be signal events, this is *not* the same as saying that the
    # DNN did not have commentary for an individual bin. For that reason, we can
    # assign bins with zero content -- assuming that *exactly* zero events can
    # represent statistical outliers due to random sampling -- to Not a Number.
    # By doing this, we ensure the color scheme is disjointed between a low
    # number of events (blue) and empty/zero (white). This is simply a graphical
    # choice.
    for x in xrange(h.GetNbinsX()):
        for y in xrange(h.GetNbinsY()):
            if not h.GetBinContent(x+1, y+1): h.SetBinContent(x+1, y+1, nan)
            pass
        pass

    # Now show the histogram
    fig, ax = plt.subplots()
    ax.set_title('DNN Result')
    if p in ['Mll', 'DPhi']:
        if p == 'Mll': ax.set_xlabel(r'$M_{\ell\ell}$', fontsize=16)
        else: ax.set_xlabel(r'$\Delta\phi$', fontsize=16)
        ax.set_ylabel('Predicted Rate')
        pass
    elif p == 'perf':
        ax.set_xlabel('Predicted Rate')
        ax.set_ylabel('Truth Value')
        pass
    hdraw = rplt.imshow(h)
    fig.colorbar(hdraw)
    if p == 'Mll':
        plt.plot([91.1876,91.1876], ax.get_ylim(), 'k')
        plt.plot(ax.get_xlim(), [0.5,0.5], 'k')
        plt.text(95, 0.95, r'$M_{\ell\ell}=m_{Z}$', fontsize=16)
        pass
    elif p == 'DPhi':
        plt.plot([0,3.5], [0.5,0.5], 'k') 
        ax.set_ylim(0, 1.05)
        pass
    else:
        plt.plot([0,1.05], [1.0,1.0], 'k')
        ax.set_xlim(0, 1.05)
        pass
    plt.show()
    #plt.savefig(os.getenv('HistDir') + '/%s.svg' % p)
    pass

f.Close()
