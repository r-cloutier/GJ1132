'''
Function to compute the sinusoidal modulation of GJ1132's light curve.
'''

def get_lc1(theta, bjd):
    As, Ac, Prot = theta
    return As*np.sin(2*np.pi*bjd/Prot) + Ac*np.cos(2*np.pi*bjd/Prot)
