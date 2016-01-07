'''
Model the lightcurve with a QP GP.
'''
from imports import *


def lnlike(params, bjd, mag, magerr):
    a_gp, l_gp, G_gp, P_gp = params
    k1 = kernels.ExpSquaredKernel(l_gp)
    k2 = kernels.ExpSine2Kernel(G_gp, P_gp)
    kernel = a_gp*k1*k2
    gp = george.GP(kernel, solver=george.HODLRSolver)
    try:
        gp.compute(bjd, magerr)
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf
    return gp.lnlikelihood(mag, quiet=True)


def lnprior(params, Plims=np.array((70, 160))):
    lna_gp, lnl_gp, lnG_gp, lnP_gp = params
    lnP_low, lnP_upp = Plims
    #if -20 < lna_gp < 20 and lnP_gp < lnl_gp < 20 and \
    #   -20 < lnG_gp < 4 and -20 < lnP_gp < 20:
    if 0 < lna_gp < 1 and lnP_gp < lnl_gp < 1e4 and \
       0 < lnG_gp < 1e2 and 0 < lnP_gp < 1:  
        return 0.0
    else:
        return -np.inf


def lnprob(params, bjd, mag, magerr):
    ll = lnlike(params, bjd, mag, magerr)
    lp = lnprior(params)
    return lp + ll


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
    lnprobs = np.zeros(0)
    p0,_,_ = sampler.run_mcmc(p0, burnin)
    # Save the lnprob to check if the chain converges
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
