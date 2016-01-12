import numpy as np


def argmax(l):
    m=-1
    id = -1
    for i, x in enumerate(l):
        if x > m:
            m = x
            id = i
    return id

def createWinnersMap(profile, trainingSet, rule):
    winners = dict()
    for i, s in enumerate(trainingSet):
        #print "finding winner for set #" + str(i)
        winners[str(s)] = rule(profile, s)
    return winners

def declareWinner(winners, delta, p_a, unknown, trainingSet):
    ## receives a preference profile, a set of candidates and the voting rule
    ## returns true if the winner is the same for all of the trainning sets.
    ## and the winner (only valid in case of 'True')
    if delta <> 0:
        return approximateWinner(winners, trainingSet, p_a, unknown, 1-delta)
    else:
        winner = winners[str(trainingSet[0])]

        if len(trainingSet) == 1:
            return True, winner

        for s in trainingSet:
            if winners[str(s)] != winner:
                return False, winner
        return True, winner

def computeSetProb(p_a, training_point, unknown_candidates):
    # Computes the joint probability of the training_point
    # conditioned on the known availability of candidates in C \ unknown_condidates
    p = 1
    for c in unknown_candidates:
        if training_point[c]:
            p = p * p_a
        else:
            p = p * 1-p_a
    return p
    
def winningProbabilities(winners, trainingSet, p_a, unknown):
    # computes the probability of each candidate to be the winner, given:
    # - the 'winners' map
    # - trainingSet - the family of sets of potential available candidates.
    # - p_a - availability probabilities.
    # - unknown - list of candidates which their availability is unknown.

    m = len(p_a)
    weights = np.zeros(m)

    for s in trainingSet:
        winner = winners[str(s)]
        weights[winner] += computeSetProf(p_a,s, unknown)

    return weights/np.sum(weights)

def approximateWinner(winners, trainingSet, p_a, unknown, alpha):
    # returns the candidate with the maximal probability of being the winner, conditioned on it having probability >= alpha
    # If none exists, returns False.
    # Input:
    # - winners - winners map
    # - trainingSet - set of Boolean vectors, denoting the possible available sets.
    # - p_a - availability probabilities.
    # - unknown - list of candidates whose availability is unknown.
    # - alpha - minimum probability

    probabilities = winningProbabilities(winners, trainingSet, p_a, unknown)
    max_candidate = argmax(probabilities)
    return (probabilities[max_candidate] >= alpha, max_candidate)
    
def one_available(availableCandidates):
    numAvailable = 0
    iAvailable = -1
    for i,x in enumerate(availableCandidates):
        if x:
            numAvailable += 1
            iAvailable = i
            #print 'Is available:',i,'. ',numAvailable,'available so far'
    if numAvailable == 1:
        # Only one available
        # Report as winner
        #print 'reporting',iAvailable,'winner because unique'
        return True,iAvailable
    else:
        return False,-1

def pluralityWinner(profile, availableCandidates):
    '''
    Computes the Plurality winner. Takes as input the preference profile (list of tuples), 
    and the set of available candidates, represented by a list of booleans.
    '''
    only_one,i = one_available(availableCandidates)
    if only_one:
        return i
    # More than one available

    m = len(profile[0])
    counts = np.zeros(m)
    for v in profile:
        i = 0
        while availableCandidates[v[i]] == False:
            i += 1
        counts[v[i]] += 1
    return argmax(counts)

def bordaWinner(profile, availableCandidates):
    '''
    Computes the Borda scoring rule winner. Takes as input the preference profile (list of tuples), 
    and the set of available candidates, represented by a list of booleans.
    '''
    only_one,i = one_available(availableCandidates)
    if only_one:
        return i
    # More than one available

    effective_m = len(availableCandidates)
    scores = np.zeros(len(profile[0]))

    for vote in profile:
        count = 0
        for i in xrange(len(vote)):
            if availableCandidates[vote[i]]:
                count += 1
                scores[vote[i]] += effective_m - count
    
    return argmax(scores)

def copelandWinner(profile, availableCandidates):
    only_one,i = one_available(availableCandidates)
    if only_one:
        return i
    # More than one available

    m = len(profile[0])
    # N(x,y) = nxy[x,y]
    nxy = np.zeros((m,m))
    
    for vote in profile:
        for i, o in enumerate(vote):
            if availableCandidates[o]:
                for j in xrange(i+1,m):
                    if availableCandidates[vote[j]]:
                        nxy[o,vote[j]] += 1
    scores = np.zeros(m)
    for x in xrange(m):
        if availableCandidates[x]:
            for y in xrange(m):
                if availableCandidates[y] and x <> y:
                    if nxy[x,y]>nxy[y,x]:
                        scores[x] +=1
                    elif nxy[x,y]<nxy[y,x]:
                        scores[x] -=1
    # print 'Copeland Scores:',scores
    return argmax(scores)

if __name__ == "__main__":
    prof = [(0,1,3,2),(3,1,2,0),(2,0,1,3)]
    prof = [(1,0,3,2),(3,0,2,1),(2,1,0,3)]
    #available = [False, True, True, True]
    available = [True, True, True, True]
    print prof
    print available
    print 'plurality:',pluralityWinner(prof,available)
    print 'borda:',bordaWinner(prof,available)
    cw = copelandWinner(prof,available)
    print 'copeland:',cw
    print
    av2 = [True, False, False, False]
    av2 = [False, False, False, True]
    prof2 = [(1,0,3,2),(3,0,2,1),(2,1,0,3)]
    print av2
    print 'plurality:',pluralityWinner(prof2,av2)
    print 'borda:',bordaWinner(prof2,av2)
    cw = copelandWinner(prof2,av2)
    print 'copeland:',cw

    big_prof = [(0, 1, 3, 7, 5, 6, 4, 2), (0, 1, 4, 7, 2, 3, 6, 5), (0, 1, 7, 4, 5, 2, 3, 6), (0, 1, 7, 5, 4, 3, 2, 6), (0, 1, 7, 6, 4, 3, 2, 5), (0, 2, 1, 6, 4, 7, 3, 5), (0, 2, 1, 7, 5, 3, 4, 6), (0, 2, 3, 5, 4, 7, 6, 1), (0, 2, 5, 1, 6, 7, 4, 3), (0, 2, 5, 3, 1, 7, 4, 6), (0, 2, 5, 4, 6, 3, 1, 7), (0, 2, 5, 7, 6, 1, 4, 3), (0, 2, 6, 1, 5, 3, 4, 7), (0, 3, 5, 7, 6, 4, 1, 2), (0, 4, 1, 3, 2, 5, 6, 7), (0, 6, 1, 2, 5, 4, 7, 3), (0, 6, 7, 5, 2, 3, 1, 4), (1, 2, 4, 3, 5, 6, 7, 0), (1, 2, 4, 6, 3, 5, 0, 7), (1, 2, 7, 6, 3, 4, 0, 5), (1, 3, 2, 7, 6, 4, 0, 5), (1, 3, 4, 7, 0, 5, 2, 6), (1, 3, 5, 6, 2, 4, 7, 0), (1, 3, 7, 0, 4, 6, 5, 2), (1, 4, 3, 6, 5, 7, 2, 0), (1, 4, 5, 6, 3, 7, 2, 0), (1, 5, 2, 3, 7, 6, 0, 4), (1, 5, 2, 4, 3, 7, 6, 0), (1, 5, 4, 6, 7, 0, 2, 3), (1, 6, 7, 0, 5, 3, 4, 2), (1, 7, 3, 2, 5, 4, 0, 6), (1, 7, 5, 3, 0, 4, 6, 2), (1, 7, 5, 3, 2, 6, 0, 4), (1, 7, 5, 3, 6, 2, 0, 4), (2, 0, 1, 5, 6, 3, 7, 4), (2, 0, 4, 7, 6, 5, 1, 3), (2, 0, 5, 7, 4, 1, 6, 3), (2, 0, 7, 5, 1, 3, 6, 4), (2, 1, 5, 7, 0, 6, 3, 4), (2, 1, 7, 0, 4, 3, 5, 6), (2, 1, 7, 4, 6, 0, 3, 5), (2, 1, 7, 5, 3, 6, 0, 4), (2, 3, 6, 1, 5, 0, 4, 7), (2, 4, 0, 1, 7, 6, 3, 5), (2, 4, 1, 3, 5, 6, 0, 7), (2, 5, 3, 7, 6, 1, 4, 0), (2, 6, 4, 0, 1, 7, 3, 5), (2, 7, 6, 3, 4, 5, 0, 1), (3, 0, 4, 5, 2, 6, 7, 1), (3, 0, 4, 6, 7, 2, 5, 1), (3, 1, 0, 2, 6, 5, 4, 7), (3, 1, 4, 0, 6, 7, 5, 2), (3, 2, 7, 4, 1, 0, 6, 5), (3, 4, 0, 1, 6, 5, 2, 7), (3, 4, 0, 6, 7, 2, 5, 1), (3, 4, 1, 2, 0, 7, 5, 6), (3, 5, 0, 1, 4, 6, 2, 7), (3, 5, 1, 4, 7, 6, 0, 2), (3, 6, 2, 4, 7, 0, 5, 1), (3, 7, 4, 0, 6, 5, 2, 1), (3, 7, 6, 2, 1, 4, 5, 0), (4, 0, 6, 1, 3, 7, 5, 2), (4, 0, 6, 5, 2, 3, 7, 1), (4, 1, 7, 2, 5, 6, 0, 3), (4, 2, 5, 1, 3, 7, 0, 6), (4, 3, 0, 5, 1, 7, 6, 2), (4, 5, 0, 3, 6, 1, 7, 2), (4, 5, 0, 6, 7, 2, 1, 3), (4, 5, 3, 6, 2, 1, 7, 0), (4, 7, 1, 0, 6, 2, 5, 3), (4, 7, 1, 5, 2, 0, 6, 3), (4, 7, 2, 0, 3, 5, 6, 1), (5, 0, 1, 3, 2, 6, 7, 4), (5, 0, 1, 4, 2, 6, 7, 3), (5, 0, 3, 1, 7, 4, 6, 2), (5, 0, 6, 3, 2, 4, 7, 1), (5, 2, 0, 6, 7, 4, 1, 3), (5, 2, 1, 4, 6, 0, 3, 7), (5, 4, 0, 2, 7, 3, 6, 1), (5, 4, 2, 7, 6, 3, 1, 0), (5, 6, 4, 3, 1, 0, 2, 7), (5, 7, 6, 3, 4, 1, 2, 0), (6, 0, 5, 7, 1, 3, 4, 2), (6, 0, 7, 2, 3, 1, 4, 5), (6, 1, 4, 2, 5, 7, 0, 3), (6, 1, 7, 2, 4, 5, 0, 3), (6, 2, 1, 0, 7, 5, 4, 3), (6, 3, 1, 7, 2, 4, 5, 0), (6, 3, 2, 0, 5, 1, 4, 7), (6, 3, 5, 4, 7, 1, 2, 0), (6, 4, 3, 2, 1, 5, 7, 0), (6, 4, 7, 2, 5, 1, 0, 3), (6, 5, 1, 3, 2, 0, 4, 7), (7, 0, 1, 4, 5, 6, 3, 2), (7, 2, 4, 5, 6, 3, 1, 0), (7, 3, 0, 2, 6, 4, 5, 1), (7, 3, 6, 1, 4, 5, 2, 0), (7, 4, 1, 2, 5, 3, 0, 6), (7, 5, 2, 4, 6, 0, 1, 3), (7, 6, 1, 5, 0, 2, 3, 4)]
    av1 = [False, True, False, False, False, False, False, False]
    avall = [True, True, True, True, True, True, True, True]

    print 'for profile with len=',len(big_prof)
    print
    print 'for av=',avall
    print 'plurality:',pluralityWinner(prof2,avall)
    print 'borda:',bordaWinner(prof2,avall)
    cw = copelandWinner(prof2,avall)
    print 'copeland:',cw
    print
    print 'for av=',av1
    print 'plurality:',pluralityWinner(prof2,av1)
    print 'borda:',bordaWinner(prof2,av1)
    cw = copelandWinner(prof2,av1)
    print 'copeland:',cw
