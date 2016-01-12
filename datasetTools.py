import sys
from itertools import permutations
from numpy.random import multinomial
from numpy.random import rand
import numpy as np

def generatePermutations(m):
    v = []
    t = permutations(xrange(m),m)
    for i in t:
        v.append(i)
    return v

cache_ktDistance = {}
def clean_ktDistance():
    cache_ktDistance = {}
    
def ktDistance(t1,t2):
    global cache_ktDistance
    ## gets two tuples and returns their Kendall-tau distance.
    k = (t1,t2)
    try:
        return cache_ktDistance[k]
    except:
        m=len(t1)
        dist = 0
        for i, v11 in enumerate(t1):
            for v12 in t1[i+1:]:
                if t2.index(v11) > t2.index(v12):
                    dist += 1
        cache_ktDistance[k] = dist
        return dist

def PrNotNorm(p, refp, disp):
    ## computes the probability of permutation p according
    ## to the Mallow's distribution: Pr[p] = disp^(ktDist(p,refp))/Z
    ## But not normalized for efficiency
    return disp**(ktDistance(p,refp))

def computeVecPr(perms,refp,disp):
    # clean_ktDistance(), not necessary to clean. Always sound
    prNotNorm = [PrNotNorm(p, refp, disp) for p in perms]
    Z = 0.0
    Ps = prNotNorm[:]
    Ps.sort() # Adding floats is not associative. Adding in order
    for p in Ps:
        Z += p

    return [p/Z for p in prNotNorm] # Now normalize

def multinomialAndReNormalize(n,P):
    while True:
        try:
            return np.random.multinomial(n, P, size=1)
        except:
            # Probably complaint on the distribution not being normalized.
            Z = 0
            Ps = P[:]
            Ps.sort() # Adding floats is not associative. Adding in order
            for p in Ps:
                Z += p
            print 'Multinomial complaint P is not normalized. (1-Sum(P)) =',(1-Z),'. Normalizing'
            for index, item in enumerate(P):
                P[index] = item / Z

def normalizer(phi,i):
    # Returns the normalizing constant needed for the repeated insertion model
    x = 0
    for j in xrange(1,i+1):
        x += phi**(j-1) 

    return x

def generateInsertionMatrix(m,phi):
    # Creates the matrix (lower-triangular) of insertion probabilities for the "repeated insertion model"
    P = []
    for i in xrange(1,m+1):
        denom = normalizer(phi,i)
        l=[]
        for j in xrange(1,i+1):
            l.append(phi**(i-j)/denom)
        P.append(l)
    return P

def generatePerm_rim(m, P):
    # Generates a permutation using the equivalent "repeated insertion model"
    # accepts as input the number of candidates, and P_{ij}-probabilities matrix
    
    insertion_vector = []
    for i in xrange(m):
            pos = np.random.multinomial(1,P[i], size=1)
            pos=pos.nonzero()[1][0]
            insertion_vector.append(pos)
    
    ranking = []
    for i in xrange(m):
            ranking.insert(insertion_vector[i],i)    
    return ranking

def generatePreferenceProfile_rim(m,n,disp):
    P = generateInsertionMatrix(m,disp)
    
    rankings = []
    for i in xrange(n):
        rankings.append(tuple(generatePerm_rim(m, P)))

    return rankings
    
def powerSet(m):
    s = [[True], [False]]
    for i in xrange(m-1):
        s1 = [list(x) for x in s]
        s2 = [list(x) for x in s]
        for i in xrange(len(s1)):
            s1[i].append(True)
            s2[i].append(False)
        s = s1 + s2

    return s

def createTrainingSet(m):
    training = powerSet(m)
    allFalse = [False for i in xrange(m)]
    training.remove(allFalse)

    return training
    
def generatePreferenceProfile_old(m, n, disp):
    # Returns a preference profile of length n
    perms = generatePermutations(m)
    refp = perms[0]
    P = computeVecPr(perms, refp, disp)
    vec = multinomialAndReNormalize(n, P)
    preferenceProfile = []
    
    for i in xrange(len(perms)):
        preferenceProfile.extend([tuple(perms[i]) for j in xrange(vec[0][i])])
            
    return preferenceProfile

def generatePreferenceProfile(m,n,disp,method):
    # Generates a preference profile given: m= number of candidates,
    # n=number of voters, disp=dispersion parameter of the Mallows model. If method=0 -- the preference profile is used by 
    #exhaustively specifying the Mallows distribution over the symmertric group S_m, and sampling n permutations.
    # If method=1 (default), the function generates n permutations using the (equivalent) repeated insertion model.
    
    if method == 0:
        profile = generatePreferenceProfile_old(m, n, disp)
    else:
        profile = generatePreferenceProfile_rim(m, n, disp)
        
    return profile

# What is this?
# Joel: This function samples a set of available candidates, represented as a list of Boolean
# values. It is not used by our code. It was written for some sanity checks.
def generateAvailableSet(m, p):
    avail = []
    v = rand(1,m)[0]
    for frac in v:
        if frac <= p:
            avail.append(True)
        else:
            avail.append(False)
    return avail
