'''
Model the lightcurve with a GP plus a sinusoid model.
'''
from imports import *
import lcmodel


def lnlike(params, bjd, mag, magerr):
    a_gp, l_gp = np.exp(params[:2])
    k1 = kernels.ExpSquaredKernel(l_gp)
    kernel = a_gp*k1
    gp = george.GP(kernel)#, solver=george.HODLRSolver)
    try:
        gp.compute(bjd, magerr)
    except (ValueError, np.linalg.LinAlgError):
        return 1e26
    model = lcmodel.get_lc1(params[2:], bjd)
    return gp.lnlikelihood(mag-model, quiet=True)


# Plims are based on the LS-periodogram
def lnprior(params, Plims=np.array((0, 300))):
    a_gp, l_gp = params[:2]
    P_low, P_upp = Plims
    As, Ac, Prot = params[2:]
    if -20 < a_gp < 20 and 8.5 < l_gp < 20 and \
       -.05 < As < .05 and -.05 < Ac < .05 and \
       P_low < Prot < P_upp:
        return 0.0
    else:
        return -np.inf


def lnprob(params, bjd, mag, magerr):
    return lnprior(params) + lnlike(params, bjd, mag, magerr)


def run_emcee_gp(params, bjd, mag, magerr, nsteps=2000, burnin=500,
                 nwalkers=100):
    '''params    = lna, lnl, A, phi, Prot'''

    # Initialize walkers in the parameter space
    ndim = len(params)
    p0=[params + 1e-8*np.random.randn(ndim) for i in range(nwalkers)]
    
    # Initialize sampler
    args = (bjd, mag, magerr)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args, a=2)

    print 'Running first Burnin (GP)...'
    t0=time.time()
    p0,_,_ = sampler.run_mcmc(p0, int(burnin/2))
    # Save the lnprob to check if the chain converges
    lnprobs = np.zeros(0)
    lnprobs = np.append(lnprobs, np.mean(sampler.lnprobability,axis=0))
    print 'Burnin acceptance fraction is %.5f' % \
        np.mean(sampler.acceptance_fraction)
    sampler.reset()

    print 'Running second Burnin (GP)...'
    p0,_,_ = sampler.run_mcmc(p0, int(burnin/2))
    lnprobs = np.append(lnprobs, np.mean(sampler.lnprobability,axis=0))
    print 'Burnin acceptance fraction is %.5f' % \
        np.mean(sampler.acceptance_fraction)
    sampler.reset()

    print 'Running MCMC (GP)...'
    p0,_,_ = sampler.run_mcmc(p0, nsteps)
    lnprobs = np.append(lnprobs, np.mean(sampler.lnprobability,axis=0))
    
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    print "Autocorrelation time:", sampler.get_autocorr_time()
    print 'MCMC took %.4e minutes'%((time.time()-t0)/60.)
    samples = sampler.chain.reshape((-1, ndim))
    return samples, lnprobs, (nwalkers, burnin, nsteps)
