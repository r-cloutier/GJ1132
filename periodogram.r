# Ryan Cloutier
#
# A script to be used by Systemic to plot the periodogram of an input
# dataset and return the periodogram and FAPs.

# Get systemic functions
source('~/linux/Systemic/R/systemic.r',chdir=TRUE)

# Initialize a kernel
k<-knew()
    
# Load noRM dataset
kremove.data(k, 'all')
k$mstar<-0.181
k$epoch<-NaN
datafile<-"~/linux/Systemic/datafiles/MEarth.vels"
kadd.data(k, c(datafile))

# Get periodogram and FAPs
#m<-kperiodogram.boot(k, per_type='res', trials=1e3,
#                     pmin=1, pmax=2e3)
m<-kperiodogram(k, per_type='res', pmin=1, pmax=2e3)
write.f(m, file='./periodogram.dat')

# Get peaks
peaks<-attr(m, 'peaks')
P<-peaks[,1]
fap<-peaks[,3]
outp<-cbind(P,fap)
write.matrix(matrix(outp,ncol=2), file='./peaks.dat')
