from numpy import *
from random import *
from matplotlib.pyplot import *

states = [0, 1, 2, 3, 4, 5, 6, 7, 8]
actions = ['G', 'D']
values = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

epsilon = 0.1 #exploration rate
alpha = 0.75 #learning rate
gamma = 0.9 #discount rate
seuil=0.01 #en dessous de cette erreur max sur une partie, on arrête l'entraînement

def rewardsystem(state):
    if state==8:
        return 0
    elif state==0:
        return -10
    else:
        return -1


def transition(state, action):
    if action == 'G':
        return state-1
    if action == 'D':
        return state+1
    

def policy(state, e=epsilon): 
    if state==8:
        return 'G'
    elif state==0:
        return 'D'
    elif random()<e:
        return choice(actions)
    else :
        return bestaction(state)


def bestaction(state):
    if values[state][1]>values[state][0]:
        return 'D'
    elif values[state][1]<values[state][0]:
        return 'G'
    else:
        return choice(actions)
    
    
def updatevalues(state, action, newstate, reward):
    if newstate == None:
        values[state]=(1-alpha)*values[state] + alpha*reward
    else:
        if action == 'D':
            values[state][1]=(1-alpha)*values[state][1] + alpha*(reward + gamma * max(values[newstate]))
        else:
            values[state][0]=(1-alpha)*values[state][0] + alpha*(reward + gamma * max(values[newstate]))            
    
def convert(action):
    if action=='D':
        return 1
    else:
        return 0

def terminal(state, k=0):
    if state == 0 or state == 8 or k>50:
        return True
    else: return False

def terminal2(n):
    if n>=20:
        return True
    else:
        return False

def erase():
    global values
    values = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

def jeu():
    state = 4 #état initial
    #print('Début')
    #print('Etat : ' + str(state) + ' ; Valeurs : ' + str(values))
    end = False
    n=0
    deltamax=0
    while end == False:
        action = policy(state)
        newstate = transition(state, action)
        reward = rewardsystem(newstate)
        oldvalue=values[state][convert(action)]
        updatevalues(state, action, newstate, reward)
        delta = abs(values[state][convert(action)]-oldvalue) #la q valeur qu'on vient de changer
        if delta>deltamax:
            deltamax=delta
        state=newstate
        n+=1
        #print('Etat : ' + str(state) + ' ; Valeurs : ' + str(values))
        end = terminal(state)
    #print('Fin')
    #print('Valeurs : ' + str(values))
    return deltamax
        
    
def train(n):
    for i in range(n):
        deltamax=jeu()
        if deltamax<seuil:
            return(i)
            break
    #plot(states, values)
    #show()

def test(): #on suit la table finale et on regarde où on arrive
    state=4
    end=False
    k=0
    while end == False:
        action=policy(state, 0)
        newstate=transition(state, action)
        state=newstate
        k+=1
        end=terminal(state, k)
    if state==8:
        return True
    else:
        return False
    
def supertest(): #le but est de tester la vitesse et la qualité de convergence de l'algorithme avec les paramètres spécifiés
    for j in range(100):
        train(100)
        test()
        erase()

def graphe():
    S=log10([0.00000000001, 0.0000000001, 0.000000001, 0.00000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
    Y=[]
    Z=[]
    for i in range(len(S)):
        global seuil
        seuil=10**S[i]
        m=0
        n=0
        print(seuil)
        for j in range(500):
            m+=train(100)
            if test()==True:
        	    n+=1
            erase()
        Y+=[m/500]
        Z+=[n/500]#nombre moyen d'itérations de convergences et pourcentage de victoire
    fig=figure()
    ax=fig.add_subplot(111)
    plot(S, Y)
    plot(S, Z)
    for xy in zip(S, Z):
        ax.annotate('%s' %xy[1], xy=xy, textcoords='data')
    for xy in zip(S, Y):
        ax.annotate('%s' %xy[1], xy=xy, textcoords='data')
    show()
    
    
