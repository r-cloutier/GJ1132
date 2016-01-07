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
    a_gp, l_gp, G_gp, P_gp = params
    P_low, P_upp = Plims
    #if 0 < a_gp and P_gp < l_gp < 1e4 and \
    #   0 < G_gp < 30 and P_low < P_gp < P_upp:  
    if P_gp < l_gp < 1e4 and \
       G_gp < 30 and P_low < P_gp < P_upp:  
        return 0.0
    else:
        return -np.inf


def lnprob(params, bjd, mag, magerr):
    ll = lnlike(params, bjd, mag, magerr)
    lp = lnprior(params)
    return lp + ll


def run_emcee_gp(params, bjd, mag, magerr, nsteps=2000, burnin=500,
                 nwalkers=100):
    '''params    = a, l, G, P'''
    # Initialize walkers in the parameter space
    ndim = len(params)
    #p0=[params + 1e-1*np.random.randn(ndim) for i in range(nwalkers)]
    a, l , G, P = params
    p0 = [[a+1e-5*np.random.randn(),
           l+1e-1*np.random.randn(),
           G+1e-4*np.random.randn(),
           P+1e-1*np.random.randn()] for i in range(nwalkers)]

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
    sampler.reset()

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
