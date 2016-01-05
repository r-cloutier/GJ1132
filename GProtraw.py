'''
Model the lightcurve with a QP GP.
'''
from imports import *

def kernel(params, bjdi, bjdj):
    a_gp, l_gp, G_gp, P_gp = np.exp(params)
    k1 = -(bjdi-bjdj)**2/(2*l_gp**2)
    k2 = -G_gp*np.sin(np.pi*(bjdi-bjdj)/P_gp)**2
    return a_gp*np.exp(k1+k2)

def lnlike(params, bjd, mag, magerr):
    # Compute covariance matrix
    N = bjd.size
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                delta = 1
            else:
                delta = 0
            K[i,j] = magerr[i]*delta + kernel(params, bjd[i], bjd[j])
    # Compute inverse of K
    Kinv = np.linalg.inv(K)
    # Compute determinant of K
    Kdet = np.linalg.det(K)
    ll = -.5*np.dot(np.dot(bjd.T, Kinv), bjd) - \
               .5*np.log(Kdet) - \
                    .5*N*np.log(2*np.pi)
    return ll


def lnprior(params, Plims=np.log(np.array((70, 160)))):
    lna_gp, lnl_gp, lnG_gp, lnP_gp = params
    lnP_low, lnP_upp = Plims
    if -20 < lna_gp < 20 and lnP_gp < lnl_gp < 20 and \
       -20 < lnG_gp < 4 and lnP_low < lnP_gp < lnP_upp:
        return 0.0
    else:
        return -np.inf


def lnprob(params, bjd, mag, magerr):
    return lnprior(params) + lnlike(params, bjd, mag, magerr)


def run_emcee_gp(params, bjd, mag, magerr, nsteps=2000, burnin=500,
                 nwalkers=100):
    '''params    = lna, lnl, lnG, lnP'''

    # Initialize walkers in the parameter space
    ndim = len(params)
    p0=[params + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    
    # Initialize sampler
    args = (bjd, mag, magerr)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)
    
    print 'Running first Burnin (GP)...'
    t0=time.time()
    p0,_,_ = sampler.run_mcmc(p0, burnin)
    # Save the lnprob to check if the chain converges
    lnprobs = np.zeros(0)
    lnprobs = np.append(lnprobs, np.mean(sampler.lnprobability,axis=0))
    print 'Burnin acceptance fraction is %.5f' % \
        np.mean(sampler.acceptance_fraction)
    print 'Burnin took %.4e minutes'%((time.time()-t0)/60.)
    #sampler.reset()

    #print 'Running second Burnin (GP)...'
    #p0,_,_ = sampler.run_mcmc(p0, int(burnin/2))
    #lnprobs = np.append(lnprobs, np.mean(sampler.lnprobability,axis=0))
    #print 'Burnin acceptance fraction is %.5f' % \
    #    np.mean(sampler.acceptance_fraction)
    #sampler.reset()

    print 'Running MCMC (GP)...'
    p0,_,_ = sampler.run_mcmc(p0, nsteps)
    lnprobs = np.append(lnprobs, np.mean(sampler.lnprobability,axis=0))
    
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    print "Autocorrelation time:", sampler.get_autocorr_time()
    print 'MCMC took %.4e minutes'%((time.time()-t0)/60.)
    samples = sampler.chain.reshape((-1, ndim))
    return samples, lnprobs, (nwalkers, burnin, nsteps)
