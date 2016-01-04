'''
Run an MCMC on an input observable(s) and use the results as 
priors for the RV model.
'''
from imports import *
import lcmodel


def lnlike(params, bjd, mag, magerr):
    a_gp, l_gp, w_gp, P_gp = np.exp(params)
    k1 = kernels.ExpSquaredKernel(l_gp)
    k2 = kernels.ExpSine2Kernel(w_gp,P_gp)
    kernel = a_gp*k1*k2
    gp = george.GP(kernel, solver=george.HODLRSolver)#, mean=np.mean(y))
    try:
        gp.compute(bjd, magerr)
    except (ValueError, np.linalg.LinAlgError):
        return 1e26
    return gp.lnlikelihood(mag, quiet=True)

def lnprior(params, Plims=np.log(np.array((50, 200)))):
    lna_gp,lnl_gp,lnw_gp,lnP_gp=params
    lnPlow, lnPupp = Plims
    if -20 < lna_gp < 20 and -20 < lnl_gp < 20 and \
       -20 < lnw_gp < 20 and lnPlow < lnP_gp < lnPupp:
        return 0.0
    else:
        return -np.inf

def lnprob(params, bjd, mag, magerr):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(params, bjd, mag, magerr)

def run_emcee_gp(params, bjd, mag, magerr, nsteps=2000, burnin=500,
                 nwalkers=32):
    '''params    = lna, lnl, lnw, lnP'''

    # Initialize walkers in the parameter space
    ndim = len(params)
    p0=[params + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    
    # Initialize sampler
    args = (bjd, mag, magerr)
    sampler=emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)#, pool=Pool())
    
    print 'Running first Burnin (GP)...'
    t0=time.time()
    p0,_,_ = sampler.run_mcmc(p0, burnin)
    # Save the lnprob to check if the chain converges
    lnprobs = np.zeros(0)
    lnprobs = np.append(lnprobs, np.mean(sampler.lnprobability,axis=0))
    print 'Burnin acceptance fraction is %.5f'%np.mean(sampler.acceptance_fraction)
    sampler.reset()

    print 'Running MCMC (GP)...'
    p0,_,_=sampler.run_mcmc(p0, nsteps)
    lnprobs = np.append(lnprobs, np.mean(sampler.lnprobability,axis=0))
    
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    print "Autocorrelation time:", sampler.get_autocorr_time()
    print 'MCMC took %.4e minutes'%((time.time()-t0)/60.)
    samples = sampler.chain.reshape((-1, ndim))
    return samples, lnprobs, (nwalkers, burnin, nsteps)
