'''
Compute the full light curve for GJ 1132 using the MEarth photometry supplied 
by Zach.
'''
from imports import *


class MEarthphotometry:
    
    def __init__(self, outsuffix=''):

        # Add obvious stuff
        self.outsuffix = outsuffix

        # Get data from Zach's file
        d = np.loadtxt('data/2MASSJ10145184-4709244_tel13_2014-2015.txt')
        self.bjd = d[:,0]
        self.mag = d[:,1]
        self.emag = d[:,2]
        self.mag0 = d[:,4]  # zero-point mag (mag0 < -0.5 are suspicious)
        self.CM = d[:,17]   # common-mode; c(t)
        self.data = d
        
        self.bjdbin()
        self.trimnbin()
        self.optimize(p0=[.07, .07, 125.])


    def bjdbin(self, binwidth=4):
        '''Bin a the timeseries in time with bins of size binwidth in 
        days.'''
        bins = np.arange(min(self.bjd), max(self.bjd), binwidth)
        nbin = bins.size
        dig = np.digitize(self.bjd, bins)
        magbin = np.array([np.mean(self.mag[dig == i]) for i in range(1,nbin)])
        emagbin = np.array([np.std(self.mag[dig == i]) for i in range(1,nbin)])
        binsv2 = np.array([np.mean([bins[i],bins[i+1]]) for i in range(nbin-1)])
        self.bjdbin  = binsv2
        self.magbin  = magbin
        self.emagbin = emagbin

    
    def trimnbin(self, binwidth=4):
        '''Trim the timeseries to only include measurements with 
        mag0 > -0.5
        Rebin the leftover data'''
        # Remove bad measurements
        good = np.where(self.mag0 > -.5)[0]
        self.bjdtrim = self.bjd[good]
        self.magtrim = self.mag[good]
        self.emagtrim = self.emag[good]
        
        # Bin the new data
        bins = np.arange(min(self.bjdtrim), max(self.bjdtrim), binwidth)
        nbin = bins.size
        dig = np.digitize(self.bjdtrim, bins)
        magbin = np.array([np.mean(self.magtrim[dig == i]) 
                           for i in range(1,nbin)])
        emagbin = np.array([np.std(self.magtrim[dig == i]) 
                            for i in range(1,nbin)])
        binsv2 = np.array([np.mean([bins[i],bins[i+1]]) 
                           for i in range(nbin-1)])
        self.bjdtrimbin  = binsv2
        self.magtrimbin  = magbin
        self.emagtrimbin = emagbin

        
    def optimize(self, p0=None):
        '''Fit a three-parameter sinusoid model to the light curve.'''
        from scipy.optimize import curve_fit
        def sinemodel(x, As, Ac, Prot):
            return As*np.sin(2*np.pi*x/Prot) + Ac*np.cos(2*np.pi*x/Prot)
        p, c = curve_fit(sinemodel, self.bjdtrim, self.magtrim, 
                         p0=p0)
        self.optmodel = sinemodel(self.bjdtrim, p[0], p[1], p[2])
        self.optresults = p
        self.optresultserr = np.sqrt(np.diag(c))

        
    def pickleobject(self):
        '''Save the complete object to a binary file.'''
        fObj = open('pickles/GJ1132_'+self.outsuffix, 'wb')
        pickle.dump(self, fObj)
        fObj.close()

    
    def quickplot(self):
        #plt.errorbar(self.bjd, self.mag, self.emag, fmt='k.')
        #plt.errorbar(self.bjdbin, self.magbin, self.emagbin, fmt='bo')
        #plt.gca().invert_yaxis()
        plt.figure(2)
        plt.errorbar(self.bjdtrim, self.magtrim, self.emagtrim, 
                     fmt='k.')
        plt.errorbar(self.bjdtrimbin, self.magtrimbin, 
                     self.emagtrimbin, fmt='bo')
        plt.plot(self.bjdtrim, self.optmodel, 'g-')
        print self.optresults, self.optresultserr
        plt.gca().invert_yaxis()
        plt.show()
        


if __name__ == '__main__':
    data = MEarthphotometry()

