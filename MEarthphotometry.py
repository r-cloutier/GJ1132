'''
Compute the full light curve for GJ 1132 using the MEarth photometry supplied 
by Zach.
'''
from imports import *
import GProt

class MEarthphotometry:
    
    def __init__(self, outsuffix='', Prot=125., 
                 thetagp=np.array((-13,6,-9,5))):  #lnsgp was ~ -12

        # Add obvious stuff
        self.outsuffix = outsuffix
        self.Prot = Prot
        self.thetagp = thetagp

        # Get data from Zach's file
        d = np.loadtxt('data/2MASSJ10145184-4709244_tel13_2014-2015.txt')
        self.bjd = d[:,0]
        self.mag = d[:,1]
        self.emag = d[:,2]
        self.mag0 = d[:,4]  # zero-point mag (mag0 < -0.5 are suspicious)
        self.CM = d[:,17]   # common-mode; c(t)
                
        self.bjdbin()
        self.trimnbin()
        

    def bjdbin(self, binwidth=4):
        '''Bin a the timeseries in time with bins of size binwidth in 
        days.'''
        bins = np.arange(min(self.bjd), max(self.bjd), binwidth)
        nbin = bins.size
        dig = np.digitize(self.bjd, bins)
        magbin = np.array([np.mean(self.mag[dig == i]) 
                           for i in range(1, nbin)])
        emagbin = np.array([np.std(self.mag[dig == i]) 
                            for i in range(1, nbin)])
        bjdbin = np.array([np.mean([bins[i],bins[i+1]]) 
                           for i in range(nbin-1)])
        good = np.where(np.isfinite(magbin))[0]
        self.bjdbin = bjdbin[good]
        self.magbin = magbin[good]        
        self.emagbin = emagbin[good]
    

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
        bjdbin = np.array([np.mean([bins[i],bins[i+1]]) 
                           for i in range(nbin-1)])
        good = np.where(np.isfinite(magbin))[0]
        self.bjdtrimbin  = bjdbin[good]
        self.magtrimbin  = magbin[good]
        self.emagtrimbin = emagbin[good]

    
    def compute_periodogram(self):
        '''Compute the Lomb-Scargle periodogram for the light curve 
        using the Systemic software.'''
        # Remove GJ1132b signal and save data
        forsystemic = np.zeros((self.bjdtrim.size, 3))
        forsystemic[:,0] = self.bjdtrim
        forsystemic[:,1] = self.magtrim
        forsystemic[:,2] = self.emagtrim
        np.savetxt('/home/ryan/linux/Systemic/datafiles/MEarth.vels',
                   forsystemic, delimiter='\t', fmt='%.6f')

        # Run R script to compute periodogram and save data to a file
        t0 = time.time()
        os.system('Rscript periodogram.r')
        print 'Computing the periodogram took %.3e minutes.'%((time.time()-t0)/60.)

        # Get periodogram data
        d = np.loadtxt('periodogram.dat')
        self.periodogramP    = d[:,0]
        self.periodogrampow  = d[:,1]
        self.periodogramfaps = d[:,2]
        d = np.loadtxt('peaks.dat')
        self.peaksP    = d[:,0]
        self.peaksfaps = d[:,1]
        # Remove large unwanted files
        os.system('rm periodogram.dat')
        os.system('rm peaks.dat')

        
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

        
    def rungp(self, nsteps=2000, burnin=500, nwalkers=40):
        '''Model the light curve with a QP GP noise model and a 
        sinusoid.'''
        samples,lnprobs,vals=GProt.run_emcee_gp(self.thetagp, 
                                                self.bjdtrimbin,
                                                self.magtrimbin, 
                                                self.emagtrimbin,
                                                nsteps=nsteps,
                                                burnin=burnin,
                                                nwalkers=nwalkers)
        self.gpsamples = samples
        self.gplnprobs = lnprobs
        self.gpvals = vals

        
    def plot_periodogram(self, label=False, pltt=False):
        '''Plot the periodogram along with the of its significant 
        peaks.'''
        plt.figure(figsize=(11,6))
        # Plot power spectrum
        plt.plot(self.periodogramP, self.periodogrampow, 'k-')
        # Highlight the actualy planet period
        plt.plot(np.repeat(self.Prot,2),
                 [0,np.ceil(max(self.periodogrampow))+1], 'b--')
        # Highlight signficant peaks and their FAPs
        for i in range(5):
            # Get power at the low FAP periods
            goodpow = np.where(np.abs(self.periodogramP-self.peaksP[i]) ==
                               np.min(np.abs(self.periodogramP-
                                             self.peaksP[i])))[0]
            plt.plot([1,2e3], np.repeat(self.periodogrampow[goodpow],2),
                     'k-', lw=.8)
            plt.text(1e2+i*1e2, self.periodogrampow[goodpow]+6e-2,
                     'FAP(P = %.3f) = %.3f'%(self.peaksP[i],
                                             self.peaksfaps[i]), fontsize=11)
        plt.xscale('log')
        plt.xlim((1,2e3))
        plt.xlabel('Period (days)')
        plt.ylabel('Power (arbitrary units)')
        plt.subplots_adjust(bottom=.12)
        if label:
            plt.savefig('plots/periodogram_'+self.outsuffix+'.png')
        if pltt:
            plt.show()
        plt.close('all')


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
    data = MEarthphotometry(outsuffix='testdummy')
    #data.optimize(p0=[.07, .07, 125.])
    #data.compute_periodogram()
    #data.plot_periodogram(pltt=1)
    data.rungp(nsteps=1000, burnin=500)
    data.pickleobject()

    
