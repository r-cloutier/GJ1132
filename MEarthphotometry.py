'''
Compute the full light curve for GJ 1132 using the MEarth photometry supplied 
by Zach.
'''
from imports import *
import lcmodel


class MEarthphotometry:
    
    def __init__(self, outsuffix='', Prot=125., thetagp=None,
                 GPonly=True):
        '''GPonly = boolean indicating whether or not to model the light
        curve as a QP GP or a GP (prbly a squared exponential) plus a 
        sine wave.''' 

        # Add obvious stuff
        self.outsuffix = outsuffix
        self.Prot = Prot
        self.GPonly = GPonly
        if thetagp == None:
            thetagp = np.exp(np.array((-11,5,.1,np.log(125),.02,.02,125)))
        else:
            thetagp = thetagp
        if self.GPonly:
            self.thetagp = thetagp[:4]
        else:
            self.thetagp = thetagp[np.array((0,1,4,5,6))]
    
        # Get MEarth photometric data
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

        
    def runqpgp(self, nsteps=2000, burnin=500, nwalkers=40):
        '''Model the light curve with a QP GP noise model and a 
        sinusoid.'''
        if self.GPonly:
            import GProt as gps 
        else:
            import GPsinerot as gps
        # First pass
        samples,lnprobs,vals=gps.run_emcee_gp(self.thetagp, 
                                              self.bjdtrimbin,
                                              self.magtrimbin, 
                                              self.emagtrimbin,
                                              nsteps=nsteps,
                                              burnin=burnin,
                                              nwalkers=nwalkers)
        self.gpsamples = samples
        self.gplnprobs = lnprobs
        self.gpvals = vals

        # Get best-fit parameters
        nparam = self.gpsamples.shape[1]
        params = np.zeros(nparam)
        for i in range(nparam):
            y,x,p = plt.hist(self.gpsamples[:,i], bins=40)
            good = np.where(y == max(y))[0][0]
            params[i] = np.mean((x[good], x[good+1]))
            plt.close('all')
        if self.GPonly:
            self.hyperparams = params
            self.mcmcparams  = None
        else:
            self.hyperparams = params[:2]
            self.mcmcparams  = params[2:]
        

    def plot_periodogram(self, label=False, pltt=False):
        '''Plot the periodogram along with the of its significant 
        peaks.'''
        plt.figure(figsize=(11,6))
        # Plot power spectrum
        plt.plot(self.periodogramP, self.periodogrampow, 'k-')
        # Highlight the actualy planet period
        plt.axvline(self.Prot, color='b', ls='--')
        # Highlight signficant peaks and their FAPs
        for i in range(5):
            # Get power at the low FAP periods
            goodpow = np.where(np.abs(self.periodogramP-self.peaksP[i]) ==
                               np.min(np.abs(self.periodogramP-
                                             self.peaksP[i])))[0]
            plt.axhline(self.periodogrampow[goodpow], color='k', ls='-', lw=.8)
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

        
    def plot_GPsummary(self, label=False, pltt=False):
        '''Make a corner plot of the GP parameter posteriors and the evolution 
        of the lnprobability = lnlikelihood + lnprior'''
        plt.figure('lnlike')
        plt.plot(self.gplnprobs, 'ko-')
        nwalkers, burnin, nsteps = self.gpvals
        plt.axvline(burnin, color='k', ls='--')
        plt.xlabel('Step Number')
        plt.ylabel('lnlikelihood')
        if label:
            plt.savefig('plots/gplnprob_'+self.outsuffix+'.png')
        
        # Next plot
        corner.corner(self.gpsamples, bins=40)
        if label:
            plt.savefig('plots/gptri_'+self.outsuffix+'.png')
        if pltt:
            plt.show()
        plt.close('all')

    
    def plot_GPmodel(self, label=False, pltt=False):
        '''Plot the lightcurve and the best-fit (most likely) GP model.'''
        if self.GPonly:
            # Compute GP model
            a_gp, l_gp, G_gp, P_gp = self.hyperparams #np.exp(self.hyperparams)
            k1 = kernels.ExpSquaredKernel(l_gp)
            k2 = kernels.ExpSine2Kernel(G_gp, P_gp)
            kernel = a_gp*k1*k2
            gp = george.GP(kernel, solver=george.HODLRSolver)
            gp.compute(self.bjdtrimbin, self.emagtrimbin)
            x = np.linspace(min(self.bjd), max(self.bjd), 3e2)
            mu, cov = gp.predict(self.magtrimbin, x)
            std = np.sqrt(np.diag(cov))
            
        else:
            # Compute GP model
            a_gp, l_gp = self.hyperparams #np.exp(self.hyperparams)
            k1 = kernels.ExpSquaredKernel(l_gp)
            kernel = a_gp*k1
            gp = george.GP(kernel, solver=george.HODLRSolver)
            gp.compute(self.bjdtrimbin, self.emagtrimbin)
            x = np.linspace(min(self.bjd), max(self.bjd), 3e2)
            #mu, cov = gp.predict(self.magtrimbin, x)
            modelsmall = lcmodel.get_lc1(self.mcmcparams, self.bjdtrimbin)
            modellarge = lcmodel.get_lc1(self.mcmcparams, x)
            samples = gp.sample_conditional(self.magtrimbin - modelsmall, 
                                            x, size=100)
            mu = np.mean(samples, axis=0) + modellarge
            std = np.std(samples, axis=0) + modellarge
        self.modelbjd = x
        self.model = mu
        self.modelerr = std

        # Plot data and model
        plt.close('all')
        plt.plot(self.bjdtrim, self.magtrim, 'k.', alpha=.1)
        plt.errorbar(self.bjdtrimbin, self.magtrimbin, self.emagtrimbin, 
                     fmt='bo')
        plt.plot(x, mu, 'g-', lw=2)
        plt.plot(x, mu+std, 'g--', lw=1.5)
        plt.plot(x, mu-std, 'g--', lw=1.5)

        plt.gca().invert_yaxis()
        plt.xlabel('BJD')
        plt.ylabel('Differential Magnitude')
        if label:
            plt.savefig('plots/gpmodel_'+self.outsuffix+'.png')
        if pltt:
            plt.show()
        plt.close('all')


    def pickleobject(self):
        '''Save the complete object to a binary file.'''
        fObj = open('pickles/GJ1132_'+self.outsuffix, 'wb')
        pickle.dump(self, fObj)
        fObj.close()



if __name__ == '__main__':
    data = MEarthphotometry(outsuffix='testdummy_gponly', GPonly=1) 

    # Try and find periodicities via LS-periodogram
    #data.compute_periodogram()
    #data.plot_periodogram(label=1, pltt=0)
    
    # Fit stuff
    data.runqpgp(nsteps=1000, burnin=200, nwalkers=36)
    data


    data.plot_GPsummary(label=1, pltt=1)
    data.plot_GPmodel(label=1, pltt=1)
    data.pickleobject()

    
