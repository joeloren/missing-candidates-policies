import argparse
import sys
import os
import glob
from numpy.random import multinomial
from numpy.random import rand
from collections import defaultdict
import numpy as np
import cPickle as cp
import math
import Queue
import random
import time
#import matplotlib.pyplot as plt

from datasetTools import *
from votingRules import *

class CNode:
    left , right, data = None, None, 0

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

    def __init__(self, data,l=None,r=None):
        # initializes the data members
        self.left = l
        self.right = r
        self.data = data

def compare_node(n1,n2):
        if n1 == n2:
                return true
        if n1.data <> n2.data:
                return false
        elif n1.data == n2.data:
                return compare_node(n1.l,n2.l) and compare_node(n1.r,n2.r)
cache = {}
def clean_cache_dp():
    global cache
    cache = {}

def report_cache_dp():
    global cache
    print 'Cache size:',len(cache)

def search_cache_dp( U, assign ):
    global cache
    fU = (frozenset(U),frozenset(assign))
    return cache[fU]

def save_cache_dp( U, assign, minCost, node ):
    global cache
    fU = (frozenset(U),frozenset(assign))
    cache[fU] = (minCost,node)

def computeProb(candidatesSet, P):
    pSet = 1
    
    for i in xrange(len(candidatesSet)):
        if candidatesSet[i] == True:
            pSet *= P[i]
        else:
            pSet *= (1-P[i])
    return pSet

def entropy(trainingSet, winners, P):
    """
    Calculates the entropy of the given training set for the winners dictionary.
    """
    val_freq     = {}
    data_entropy = 0.0

    # finding all the winners for the given training set
    winnersSetsDict = defaultdict(list)
    for candidateSet in trainingSet:
        w = winners[str(candidateSet)]
        winnersSetsDict[w].append(candidateSet)
        
    data_entropy = 0
    for w in winnersSetsDict.keys():
        pw = 0
        for candidateSet in winnersSetsDict[w]:
            pw += computeProb(candidateSet, P)
        data_entropy += -pw * math.log(pw,2)
    return data_entropy

def gain(trainingSet, winners, candidate, P):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    val_freq       = {}
    subset_entropy = 0.0
    currentEntropy = entropy(trainingSet, winners, P)

    aSets = [candidatesSet for candidatesSet in trainingSet if candidatesSet[candidate] == True]
    uSets = [candidatesSet for candidatesSet in trainingSet if candidatesSet[candidate] == False]

    # computing conditional probability of each set
    totalProb = 0.0
    for candidatesSet in trainingSet:
        totalProb += computeProb(candidatesSet, P)
        
    aProb = 0.0
    for candidatesSet in aSets:
        aProb += computeProb(candidatesSet, P)
    aProb = aProb / totalProb
    
    uProb = 0.0
    for candidatesSet in uSets:
        uProb += computeProb(candidatesSet, P)
    uProb = uProb / totalProb
    
    newEntropy = aProb * entropy(aSets, winners, P) + uProb * entropy(uSets, winners, P)

    return (currentEntropy - newEntropy)

def policyMyopicRec(m, training, winners, delta, P, U):
    """
    A recursive function for creating the decision tree using the myopic heuristic
    Obtains the number of agents, training set,
    winners that map sets to winning candidates, availability vector P, and set of non-queried candidates U
    """
    pure,w = declareWinner(winners, delta, P, U, training)
    node = CNode(w)

    if not pure:
        U_copy = set(U)
        #igVec = np.zeros(m)
        maxId = 0
        maxGain = -1
        for c in U:
            ig =  gain(training, winners, c, P)
            if maxGain < ig:
                maxGain = ig
                maxId = c
        node.data = maxId
        U_copy.remove(maxId)
                
        NTraining = [s for s in training if s[node.data] == False]
        YTraining = [s for s in training if s[node.data] == True]

        node.left = policyMyopicRec(m, YTraining, winners, delta, P, U_copy)
        node.right = policyMyopicRec(m, NTraining, winners, delta, P, U_copy)
        
    return node

def policyMyopic(profile, m, P,C, trainingSet, winners, delta):
    res = policyMyopicRec(m, trainingSet, winners, delta, P, set(range(m)))
    cost = calcExpectedCost(res, P, C)
    return (cost, res)

def policyRandomRec(m, training, winners, delta, P, U):
    """
    A recursive function for creating the decision tree
    choosing randomly the candidate to query.using the myopic heuristic
    Obtains the number of agents, training set,
    winners that map sets to winning candidates, availability vector P, and set of non-queried candidates U
    """
    pure,w = declareWinner(winners, delta, P, U, training)
    node = CNode(w)

    if not pure:
        selectedId = random.choice(U)
        node.data = selectedId
        U_copy = set(U)
        U_copy.remove(selectedId)
        U_copy = list(U_copy)
                
        NTraining = [s for s in training if s[node.data] == False]
        YTraining = [s for s in training if s[node.data] == True]

        node.left = policyRandomRec(m, YTraining, winners, delta, P, U_copy)
        node.right = policyRandomRec(m, NTraining, winners, delta, P, U_copy)
        
    return node

def policyRandom(profile, m, P,C, trainingSet, winners, delta):
    res = policyRandomRec(m, trainingSet, winners, delta, P, range(m))
    cost = calcExpectedCost(res, P, C)
    return (cost, res)


# Invariant: {0, ..., m } == U \union { c | abs(c+1) \in assign }
# c moves from U to assign, but increased +1 to keep representation uniq.
# assign is useful for caching.
def policyDPRec(profile, training, winners, delta, P, C, U, assign):
    ## Receives a set 'trainning' of candidates, probability vector 'p' for any candidate to be available, cost vector 'c' for querying candidates
    ##Returns
    try:
        return search_cache_dp( U, assign )
    except:
        pass
    pure,w = declareWinner(winners, delta, P, U, training)    
    node = CNode(w)
    minCost = 0
    if not pure:
        minCost = np.inf
        minCandidate = None
        minYNode, minNNode = None, None
        #print "finding min cost query over unknown candidates: " + str(U)
        for c in U:
            # New U
            NU = set(U)
            NU.remove(c)

            YTraining = [s for s in training if s[c] == True]
            assign_y = assign[:]
            assign_y.append(c+1) # representation in assign with polarity. +1 because 'c' can be 0.
            Ycost, Ynode = policyDPRec(profile, YTraining, winners, delta, P, C, set(NU), assign_y)

            NTraining = [s for s in training if s[c] == False]
            assign_n = assign[:]
            assign_n.append(-(c+1)) # representation in assign with polarity. +1 because 'c' can be 0.
            Ncost, Nnode = policyDPRec(profile, NTraining, winners, delta, P, C, set(NU), assign_n)

            cost = C[c] + P[c]*Ycost + (1-P[c])*Ncost
            if cost < minCost:
                minCost = cost
                minYNode = Ynode
                minNNode = Nnode
                minCandidate = c
        node = CNode(minCandidate,minYNode,minNNode)
        #print "min cost of " + str(minCost) + " evaluated at candidate " + str(minCandidate)
    save_cache_dp( U, assign, minCost, node )
    return (minCost, node) 

def policyDP(profile, m, P,C, trainingSet, winners, delta):
    clean_cache_dp()
    cost, res = policyDPRec(profile, trainingSet, winners, delta, P, C, set(range(m)), [])
    report_cache_dp()
    return (cost, res)

# def inorder(node):
#     print node.data
#     inorder(node.left)
#     print node.data
#     inorder(node.right)
#     return

def calcCost(perm, T, C, available):
# Calculates the cost of determining the winner for the given boolean array 'available' of available candidates
    cost = 0
    node = T
    while not node.is_leaf():
        cost += C[node.data]
        if available[node.data] == True:
            node = node.left
        else:
            node = node.right
    return cost

def createAvailableSet(m, P):
    # creates a boolean list of available candidates according to the vector of bernoulli probabilities 'P'
    available = [np.random.rand()<= P[i] for i in xrange(m)]
    return available

def calcExpectedCost(T, P, C):
    if T.is_leaf():
        return 0
    return C[T.data] + P[T.data]*calcExpectedCost(T.left,P,C) + (1-P[T.data])*calcExpectedCost(T.right,P,C)

# deprecated
def calcAvgCost(perm, T, P, C, m, training):
    cost=0
    
    for i, available in enumerate(training):
        cost += P[i] * calcCost(perm,T,C,available)
    return cost

def get_datafile(name):
    f = open(name,'r')
    data = cp.load(f)
    f.close()
    return data

def load_data():
    global rule, ruleName, alg, algName, filenamePreferenceProfile, filenameWinners, filenameData
    global preferenceProfileAll, winnersAll
    
    print "Using data from files"
    try:
        preferenceProfileAll = get_datafile(filenamePreferenceProfile)
        print 'Loaded profiles'
    except:
        print "Failed to load profile data-set:", filenamePreferenceProfile
        print "Call the program again with option for generating data. -h for help"
        sys.exit(1)
    try:
        winnersAll = get_datafile(filenameWinners)
        print 'Loaded winners'
    except:
        print "Failed to load winner for data-set:", filenameWinners
        print "Call the program again with option for generating data. -h for help"
        sys.exit(1)
    print "Done loading files"
    print

def gnode(n):
    return 'node'+str(n)

def printTree(T,i):
    global filenameData
    
    f = open(filenameData+"-p"+str(i)+".dot", 'w')
    print >> f, "Digraph G {"

    q = Queue.Queue()
    num = 0
    q.put((T,num))
    while not q.empty():
        (T,n) = q.get()
        print >> f, gnode(n),
	print >> f, '[ label="',T.data,'"];'
        if T.left <> None:
            num += 1
            print >> f, gnode(n), '->', gnode(num),';'
            q.put((T.left,num))
        if T.right <> None:
            num += 1
            print >> f, gnode(n), '->', gnode(num),'[style=dotted];'
            q.put((T.right,num))
        
    print >> f, "}"
    f.close()

def run_alg(trainingSet):
    global rule, ruleName, alg, algName, filenamePreferenceProfile, filenameWinners
    global preferenceProfileAll, winnersAll

    print "Voting rule:", ruleName
    print "Algorithm for creating the decision tree:", algName
    
    C = np.ones(parameters.m)
    P = parameters.ap * np.ones(parameters.m)
    print "Probability vector: " + str(P)
    print "Cost vector: " + str(C)
    totalCost = 0
    totalEvaluatedCost = 0
    data = np.zeros(parameters.populations)
    for i in xrange(parameters.populations):
        preferenceProfile = preferenceProfileAll[i]
        winners = winnersAll[i]
        print "Generating the decision tree for profile",i
        cost, T = alg(preferenceProfile, parameters.m, P, C, trainingSet, winners, parameters.delta)
        print "Expected cost =", cost # now generating random available sets and computing costs"
                                      #evaluatedCost = calcAvgCost(preferenceProfile, T, P, C, parameters.m, trainingSet)
                                      #print "Average cost over", parameters.repeats, "randomly generated available candidate sets:", evaluatedCost
        print
        totalCost += cost
        totalEvaluatedCost += cost
        data[i] = cost
        printTree(T,i)
    print "Overall average cost for " + str(parameters.populations) + " populations: " + str(totalCost / parameters.populations)
    print 'data-plot:',data
    f = open(filenameData+'.dat', 'w')
    cp.dump(data, f)
    f.close()

    #print "Overall average evaluated cost for " + str(parameters.populations) + " populations: " + str(totalEvaluatedCost / parameters.populations)

def generate_data(trainingSet):
    global rule, ruleName, alg, algName, filenamePreferenceProfile, filenameWinners, filenameData
    global preferenceProfileAll, winnersAll
    
    # Only overwrites data if not generated
    try:
        preferenceProfileAll = get_datafile(filenamePreferenceProfile)
        print "Using an existing data set file", filenamePreferenceProfile
        print "You may delete it and run the program again."
    except:
        preferenceProfileAll = []
        for i in xrange(parameters.populations):
            print "profile #" + str(i)
            preferenceProfile = generatePreferenceProfile(parameters.m,parameters.n,parameters.disp, parameters.ppg)
            preferenceProfileAll.append( preferenceProfile )
        f = open(filenamePreferenceProfile, 'w')
        cp.dump(preferenceProfileAll, f)
        f.close()
        print "Done! Saved", filenamePreferenceProfile, "cPickle files"

        # New Data. Delete old winners file. Better to delete than to make a mistake.
        for f in glob.glob("dataWinners" + suffix + "*.cp"):
            print 'To delete', f
            os.remove(f)

    try:
        winnersAll = get_datafile(filenameWinners)
        print "Using an existing winner data set file", filenameWinners
        print "You may delete it and run the program again."
    except:
        winnersAll = []
        for i in xrange(parameters.populations):
            preferenceProfile = preferenceProfileAll[i]
            # HECTOR: Joel: please check this old comment. I think it was inexact.
            # print "generating map of winning candidates for different available sets"
            print "generating map of winning candidates for profile #"+str(i)
            winnersAll.append( createWinnersMap(preferenceProfile, trainingSet, rule) )
        f = open(filenameWinners, 'w')
        cp.dump(winnersAll, f)
        f.close()
        print "Done! Saved", filenameWinners, "cPickle files"

    print "Run with the generate data option to run the algorithm"

def mult_sorted(v):
    v.sort()
    # Multiply in increasing order
    res = 1.0
    for item in v:
        res *= item
    return res

def sum_sorted(v):
    v.sort()
    # Sum in increasing order
    res = 0.0
    for item in v:
        res += item
    return res

def plot_whatif(trainingSet):
    global rule, ruleName, alg, algName, filenamePreferenceProfile, filenameWinners, filenameData

    debug_it = False
    if debug_it:
        for i, prof in enumerate(preferenceProfileAll):
            print 'Prof #',i,prof
        print 'End of profiles'
        print

    allAvailable = [True] * parameters.m
    delta = 1e-15
    xv = []
    yv = []
    yvSame = []
    yvNumSame = []
    yvMass = []
    yvNumMass = []
    if debug_it:
        end,step = 0.0012,0.0001
    else:
        end,step = 1.0,0.01
    for ap in np.arange(delta,end+(10*delta),step):
        cacheAV = {}
        if ap > 1:
            ap = 1.0
        v_total = []
        v_totalSame = []
        v_numSame = []
        v_totalMass = []
        v_numMass = []
        P = ap * np.ones(parameters.m)
        for i, prof in enumerate(preferenceProfileAll):
            winners = winnersAll[i]
            w = winners[str(allAvailable)] # first winner
            # w is available for sure
            TS = [s for s in trainingSet if s[w] == True]

            v_pSame = []
            v_massTS = []
            for av in TS:
                av_t = tuple(av)
                try:
                    prob_av = cacheAV[av_t]
                except:
                    v_prob_av = []
                    for j, o in enumerate(av):
                        if o:
                            v_prob_av.append(P[j])
                        else:
                            v_prob_av.append(1-P[j])
                    prob_av = mult_sorted(v_prob_av)
                    cacheAV[av_t] = prob_av
                    
                if winners[str(av)] == w:
                    # Winner coincides
                    v_pSame.append(prob_av)
                    if debug_it:
                        if ap == delta:
                            print 'Is the same:',prob_av,av
                elif ap == delta:
                    if debug_it:
                        print 'NOTthe same:',prob_av,av
                v_massTS.append(prob_av)
            pSame = sum_sorted(v_pSame)
            massTS = sum_sorted(v_massTS)
            v_total.append(pSame/massTS)
            if debug_it:
                v_totalSame.append(pSame)
                v_totalMass.append(massTS)
                v_numSame.append(len(v_pSame))
                v_numMass.append(len(v_massTS))
                if ap == delta:
                    print 'ALL THE SAME (#%d). Winner = %d. numSame=' % (i,w),len(v_pSame),'.  numMass=',len(v_massTS)
                    print 'psame=',pSame,'. massTS-psame=',massTS-pSame,'.  massTS=',massTS, '. total +=',pSame/massTS
        # print 'pa =',ap,'. prob =',total/len(preferenceProfileAll)
        xv.append(ap)
        yv.append(sum_sorted(v_total)/len(preferenceProfileAll))
        if debug_it:
            yvSame.append(sum_sorted(v_totalSame)/len(preferenceProfileAll))
            yvMass.append(sum_sorted(v_totalMass)/len(preferenceProfileAll))
            yvNumSame.append(sum_sorted(v_numSame)/len(preferenceProfileAll))
            yvNumMass.append(sum_sorted(v_numMass)/len(preferenceProfileAll))
    f = open(filenameData+'-wf.dat', 'w')
    cp.dump((xv,yv),f)
    f.close()
    if debug_it:
        f = open(filenameData+'-wf-same.dat', 'w')
        cp.dump((xv,yvSame),f)
        f.close()
        f = open(filenameData+'-wf-mass.dat', 'w')
        cp.dump((xv,yvMass),f)
        f.close()
        i = len(xv) - 4 
        print 'SAMPLE',xv[i], yv[i],yvSame[i],yvMass[i]
        print 'num',yvNumSame[i],yvNumMass[i]

def main():
    print "Creating training set"
    trainingSet = createTrainingSet(parameters.m)
    if parameters.whatif:
        load_data()
        plot_whatif(trainingSet)
    else:
        if parameters.generateData:
            generate_data(trainingSet)
        else:
            load_data()
            run_alg(trainingSet)
        
    
parser = argparse.ArgumentParser()

# Uncomment 'required' if we want the parameter to be set by default

parser.add_argument('-n', type = int, #required,
                    default = 100, help="size of preference profile (i.e. number of agents)" )

parser.add_argument('-m', type = int, #required,
                    default = 8, help="size of candidate set" )

parser.add_argument('-disp', type = float, #required,
                    default = 0.7, help="dispersion parameter for Mallow's model. Float between 0 and 1")

parser.add_argument('-ap', type = float, #required,
                    default = 0.5, help="probability of each candidate being available" )

# parser.add_argument('-rep', dest='repeats', type = int, required,
#                     default = 100, help="number of sets of available candidates the cost should be computed for" )

parser.add_argument('-pop', dest='populations', type = int, #required,
                    default = 1, help="populations" )

parser.add_argument('-vr', dest='votingRule', type = int, choices = [0,1,2], #required,
                    default = 0, help=" voting rule to be used. 0 = plurality, 1 = Borda, 2 = Copeland, others to be implemented. " )
                    # (dictionary in main part specifieds the function to be used accordingly )"

parser.add_argument('-gd', dest='generateData', action="store_true", #required,
                    default = False, help="For generating data with canonical names. Omit this option to run the algorithm (but it will complaint if does have data)")

parser.add_argument('-method', dest='alg', type = int, choices=[0,1,2], #required,
                    default = 0, help="For selecting an algorithm. 0 = Dynamic programming (default), 1 = Myopic (C4.5), 2 = random")

parser.add_argument('-ppg', type = int, choices=[0,1], #required,
		    default = 1, help="Preference profile generation method. 0 = Exhaustive enumeration, 1 = Repeat insertion model (default)")

parser.add_argument('-delta', type = float, #choices=[0,1], #required,
		    default = 0, help="Policy will return winners with probablity 1-delta. Default = 0, meaning only guaranteed winners")

parser.add_argument('-whatif', action="store_true", #required,
                    default = False, help="Assuming 1st winner is a winner, plot the probability of keep being the winner after queries. Ignore parameter -ap")


rulesDict = {0 : (pluralityWinner, "Plurality"), 1 : (bordaWinner, "Borda"), 2 : (copelandWinner, "Copeland")}
algsDict = {0 : (policyDP, "Dynamic programming"), 1 : (policyMyopic, "Myopic algorithm"), 2 : (policyRandom, "Random algorithm")}

if __name__ == "__main__":
    t1 = time.time()
    parameters = parser.parse_args()
    if parameters.generateData and parameters.whatif:
        print "Cannot generatedData and plot whatif. Choose one: -gd or -whatif"
        sys.exit(1)

    # Setting global vars
    print 'Parameters:',parameters
    rule, ruleName = rulesDict[parameters.votingRule]
    alg, algName = algsDict[parameters.alg]

    suffix = "-m=" + str(parameters.m) + "-n=" + str(parameters.n) + "-disp=" + str(parameters.disp) + "-popsize=" + str(parameters.populations)
    filenamePreferenceProfile = "preferenceProfile" + suffix + ".data"
    suffix2 = suffix + "-vr="+str(parameters.votingRule)
    filenameWinners = "winners" + suffix2 + ".data"
    suffix3 = suffix2 + "-ap="+str(parameters.ap) + "-method="+str(parameters.alg)
    filenameData = "data" + suffix3 

    main()
    print 'finishing with Parameters:',parameters
    print
    print '***************************************************************************'
    print '***************************************************************************'
    print
    t2 = time.time()
    print "Elapsed time: " + str((t2-t1)) + " seconds."
