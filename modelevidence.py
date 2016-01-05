'''
Estimate the model evidence for a QP GP model and a GP plus sinusoid 
for the input light curve.
'''
from imports import *
from MEarthphotometry import *

class modelevidence:

    def __init__(self, 
                 file0='pickles/GJ1132_testdummy_gpplussine_lc1',
                 file1='pickles/GJ1132_testdummy_gponly'):
        
        # Add stuff
        self.files = np.array((file0, file1))
        self.get_data()
        self.compute_integrated_lnlike()
        self.bayesK_0over1 = self.marglike0/self.marglike1
        self.bayesK_1over0 = self.marglike1/self.marglike0
        self.draw_conclusion()

        # Save
        self.pickleobject()

        
    def get_data(self):
        '''Read in the pickles and save the MCMC posteriors.'''
        for i in range(2):
            # Readin data
            fObj = open(self.files[i], 'rb')
            data = pickle.load(fObj)
            fObj.close()

            # Get posteriors
            if i == 0:
                self.samples0 = data.gpsamples
            else:
                self.samples1 = data.gpsamples

            # Get other required data
            self.bjdtrimbin  = data.bjdtrimbin
            self.magtrimbin  = data.magtrimbin
            self.emagtrimbin = data.emagtrimbin


    def compute_integrated_lnlike(self, N=int(5e3)):
        '''For each model:
        for i->N
           draw model parameters from the posteriors and compute models
           use gp.lnlikelihood to compute the lnlikelihood of that model
           (dont care about priors because they're uniform)
        integrate over all parameters to get the marginalized likelihood
        for each model.'''

        def compute_lnlike(theta):

            def Glnlike(y, yerr, mu):
                return -.5*np.sum(((y-mu)/yerr)**2)

            if theta.size == 5:
                agp, lgp = np.exp(theta[:2])
                As, Ac, Prot = theta[2:]
                gp = george.GP(agp*kernels.ExpSquaredKernel(lgp))
                gp.compute(self.bjdtrimbin, self.emagtrimbin)
                model = lcmodel.get_lc1(theta[2:], self.bjdtrimbin)
                #return gp.lnlikelihood(self.magtrimbin-model, quiet=1)
                try:
                    samples = gp.sample_conditional(self.magtrimbin - model, 
                                                    self.bjdtrimbin, size=100)
                    mu = np.mean(samples, axis=0) + model
                except:
                    mu = np.zeros(self.magtrimbjd.size)
                return Glnlike(self.magtrimbin, self.emagtrimbin, mu)
                
            elif theta.size == 4:
                agp, lgp, Ggp, Pgp = np.exp(theta)
                k1 = kernels.ExpSquaredKernel(lgp)
                k2 = kernels.ExpSine2Kernel(Ggp, Pgp)
                gp = george.GP(agp*k1*k2)
                gp.compute(self.bjdtrimbin, self.emagtrimbin)
                #return gp.lnlikelihood(self.magtrimbin, quiet=1)
                try:
                    samples = gp.sample_conditional(self.magtrimbin, 
                                                    self.bjdtrimbin, 
                                                    size=100)
                    mu = np.mean(samples, axis=0)
                except:
                    mu = np.zeros(self.magtrimbin.size)
                return Glnlike(self.magtrimbin, self.emagtrimbin, mu)


        # Draw parameters from posteriors and compute the lnlike
        lnlike = np.zeros((N,2))
        for i in range(N):
            # Draw model parameters
            theta0 = np.zeros(5)
            for j in range(5):
                theta0[j] = np.random.choice(self.samples0[:,j])
            theta1 = np.zeros(4)
            for j in range(4):
                theta1[j] = np.random.choice(self.samples1[:,j])

            # Compute likelihood of these model parameters
            lnlike[i,0] = compute_lnlike(theta0)
            lnlike[i,1] = compute_lnlike(theta1) 

        # Save lnlike distribution
        self.lnlike = lnlike

        # Integrate to get marginalized likelihood (linear units!!)
        from scipy.integrate import simps
        self.marglike0 = simps(np.exp(self.lnlike[:,0]), axis=0)
        self.marglike1 = simps(np.exp(self.lnlike[:,1]), axis=0)

    
    def draw_conclusion(self):
        '''Based on the ratio of marginalized likelihoods, return an 
        interpretation.
        0 = GP + sinusoid
        1 = QP GP'''
        ratio = self.bayesK_0over1
        if ratio <= 1e-2:
            interp = 'A QP GP is decisively favoured.'
        elif 1e-2 < ratio <= 10**(-1.5):
            interp = 'A QP GP is very strongly favoured.'
        elif 10**(-1.5) < ratio <= 1e-1:
            interp = 'A QP GP is strongly favoured.'
        elif 1e-1 < ratio <= 10**(-.5):
            interp = 'A QP GP is substantially favoured.'
        elif 10**(-.5) < ratio <= 10**(.5):
            interp = 'Neither model is favoured.'
        elif 10**(.5) < ratio <= 1e1:
            interp = 'A GP + sinusoid is substantially favoured.'        
        elif 1e1 < ratio <= 10**(1.5):
            interp = 'A GP + sinusoid is strongly favoured.'
        elif 10**(1.5) < ratio <= 1e2:
            interp = 'A GP + sinusoid is very strongly favoured.'        
        else:
            interp = 'A GP + sinusoid is decisively favoured.'
        self.interpretation = interp

    
    def pickleobject(self):
        '''Save the object via pickling.'''
        fObj = open('pickles/modelevidence', 'wb')
        pickle.dump(self, fObj)
        fObj.close()


if __name__ == '__main__':
    data = modelevidence()
