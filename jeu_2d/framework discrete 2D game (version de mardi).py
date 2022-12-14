from numpy import *
from matplotlib.pyplot import *
from random import *
from tkinter import *
import time


class environnement():
    """Une instance de cette classe est un environnement, un terrain de jeu, tout ce qui enveloppe le reste."""
    
    def __init__(self, n, rewarded_states=[], rewards=[], terminal_states=[], forbidden_states=[], start_point=[0, 0], max_plays = 30, move_penalty = -1):

        self.states=[[] for k in range(n)] #la matrice des états de l'environnement = cases
        self.len=n #la taille de la grille (carrée)
        self.start = start_point #vecteur donnant la position de départ
        self.rewards=rewards
        self.max_plays = max_plays
        self.move_penalty = move_penalty

        self.rs=rewarded_states #liste de vecteurs
        R=rewards #liste de scalaires
        #le choix de créer une liste d'états récompensés et une liste des récompenses
        #vient du fait qu'en général il y a peu d'état récompensés et donc
        #il serait lourd d'avoir juste une grosse matrice de récompense quasi-nulle

        self.ts=terminal_states #les états sur lesquels le jeu s'arrête (liste de vecteurs)
        self.fs=forbidden_states #les états interdits = murs ... (liste de vecteurs)
        
        #création des états (objets)
        for i in range(n):
            for j in range(n):
                self.states[i]+=[state(self, i, j)]


        self.start = self.states[start_point[0]][start_point[1]] #état de départ


        def color(self, state):
            if state.isterminal:
                return 'red'
            if state.isrewarded:
                return 'yellow'
            if state.isforbidden:
                return 'black'
            if state == self.start:
                return 'blue'
            return None

            
        self.window = Tk()
        self.size = int(600/self.len)*self.len
        self.step = self.size / self.len
        self.canvas = Canvas(self.window, height=self.size+1, width=self.size+1)
        for i in range(n):
            for j in range(n):
                s=self.states[i][j]
                self.canvas.create_rectangle(2+j*self.step, 2+i*self.step, 2+(j+1)*self.step, 2+(i+1)*self.step, fill = color(self, s))


        for i in range(self.len+1):
            self.canvas.create_line(i*self.step+2, 2, i*self.step+2, self.size+2)
        for i in range(self.len+1):
            self.canvas.create_line(2, i*self.step+2, self.size+2, i*self.step+2)
        self.canvas.pack()
        #dans un canvas tkinter, les coordonnées dessinables vont de 2 à n+1 où n est une dimension

class state():
    """Une instance de cette classe est un état de l'environnement, elle est donc générée automatiquement par un environnement
    relativement à lui-même. Un objet état contient toutes les informations nécessaire relatives à lui-même : position sur la grille,
    actions possibles, récompense, terminalité."""
    
    def __init__(self,env, x, y):

        self.x=x
        self.y=y


        self.isrewarded=False
        self.reward=0
        self.isterminal=False
        self.isforbidden=False

        #actualisation des attributs selon les paramètres de l'environnement
        if [x, y] in env.rs:
            self.isrewarded=True
            indice=(env.rs).index([x, y])
            self.reward=env.rewards[indice]
            
        if [x, y] in env.ts:
            self.isterminal=True
            
        if [x, y] in env.fs:
            self.isforbidden=True

            

        self.possible_actions=['Left', 'Right', 'Up', 'Down']
        
        #élimination des mouvements au bord selon coordonnée Y
        if y==0:
            self.possible_actions.remove('Left')
        elif y==env.len-1:
            self.possible_actions.remove('Right')
        #élimination des mouvements au bord selon coordonnée X
        if x==0:
            self.possible_actions.remove('Up')
        elif x==env.len-1:
            self.possible_actions.remove('Down')

        #élimination des mouvements selon les états interdits adjacents
        if [x, y-1] in env.fs:
            self.possible_actions.remove('Left')
        if [x, y+1] in env.fs:
            self.possible_actions.remove('Right')
        if [x-1, y] in env.fs:
            self.possible_actions.remove('Up')
        if [x+1, y] in env.fs:
            self.possible_actions.remove('Down')

E=environnement(n=7, rewarded_states=[[0,0],[0,1],[0,6],[1,1],[1,5],[2,4],[6,4],[6,5]], terminal_states=[[0,1],[0,6],[1,1],[2,4]], rewards=[100, -1000, -1000, -1000, 10, -1000, 2, 2], start_point=[5, 1])    

class player():
    """Une instance de cette classe simule à la fois un joueur du jeu et la partie qu'il joue.
Dépend donc d'un objet environnement."""

    def __init__(self, environnement, family, epsilon):
        self.state = environnement.start #on commence sur l'état de départ
        self.previous_state=None
        self.score = 0
        self.isover = False #est-ce que la partie est finie
        self.deltamax = 0 #la valeur abs de la correction maximale effectuée sur la QTable pendant la partie

        self.fam = family
        self.e = epsilon
        self.env = environnement

        self.movements = 0
        
    def play(self, learning=True):
        """Effectue une action"""
        if random()<self.e and learning:
            order = choice(self.state.possible_actions)
        else:
            order=self.best_action()

        self.previous_state = self.state
        self.move(order)
        
        self.score+=self.state.reward + self.env.move_penalty

        old_value = self.fam.QTable[self.previous_state.x][self.previous_state.y][self.fam.nombre(order)]
        self.fam.update_QTable(self.previous_state, order, self.state, self.state.reward + self.env.move_penalty)
        if abs(old_value - self.fam.QTable[self.previous_state.x][self.previous_state.y][self.fam.nombre(order)]) > self.deltamax :
            self.deltamax = abs(old_value - self.fam.QTable[self.previous_state.x][self.previous_state.y][self.fam.nombre(order)])
        
        if self.state.isterminal == True:
            self.isover = True
        self.movements += 1

             
    def best_action(self): #best action according to QTable of family
        liste = []
        for action in self.state.possible_actions:
            liste+=[self.fam.QTable[self.state.x][self.state.y][self.fam.nombre(action)]]
        maxi = 0
        ind_list = []
        for i in range(len(liste)):
            if liste[i]==max(liste):
                ind_list+=[i]
        return self.state.possible_actions[choice(ind_list)]
    
        
    
    def move_left(self):
        self.state=self.env.states[self.state.x][self.state.y-1]
    def move_right(self):
        self.state=self.env.states[self.state.x][self.state.y+1]
    def move_down(self):
        self.state=self.env.states[self.state.x+1][self.state.y]
    def move_up(self):
        self.state=self.env.states[self.state.x-1][self.state.y]
    def move(self, order):
        if order=='Left':
            self.move_left()
        elif order=='Right':
            self.move_right()
        elif order=='Up':
            self.move_up()
        elif order=='Down':
            self.move_down()
    
                                   

class family():
    """Un objet de cette classe va être un superobjet qui contiendra des informations
sur un entraînement réalisé avec une série de parties. La méthode train() va instancier
des objets player (= va créer des parties) et les faire jouer jusqu'a un
état terminal (=va jouer les parties) jusqu'à atteindre les critères de convergence.
De là, on pourra accéder à la QTable obtenue suite à cet entraînement, et faire
jouer des parties sans la modifier pour observer le comportement en régime établi,
ou reprendre l'entraînement.
C'est un tel objet qui va être repésenté graphiquement, car il dépend d'un environnement
qu'on pourra représenter, et contient des players qui pourront être représentés
successivement, ainsi qu'une QTable évolutive."""

    def __init__(self, env = E, alpha = 0.75, gamma = 0.9): #notre famille doit avoir un QTable limite (dépend de l'environnement et des paramètres alpha et gamma)
        #                                   unique vers laquelle on tend en entraînant
        self.env = env
        self.QTable=zeros((E.len, E.len, 4)) #la QTable de la famille , prenant en paramètre les coordonnées puis l'action
        self.a = alpha
        self.g = gamma

    def birth(self, epsilon=0.1):
        self.P = player(self.env, self, epsilon)

    def train(self, epsilon = 1, N = 200, stop = True , threshold = 0.001): #on train sur plusieurs parties en explorant d'une certaine façon
        """Fait jouer des joueurs successifs et arrête l'entraînement soit quand le deltamax de la dernière partie jouée est inférieur au seuil (si stop est true)
    soit après avoir fait jouer N joueurs"""
        self.birth(epsilon)
        deltamax = math.inf
        n=0
        
        while (stop and deltamax>threshold and n<N) or (stop==False and n<N):

            #on joue la partie
            o = self.env.canvas.create_oval(self.P.state.y*self.env.step -10 + self.env.step/2, (self.P.state.x)*self.env.step-10 + self.env.step/2, self.P.state.y*self.env.step+10 + self.env.step/2, (self.P.state.x)*self.env.step+10 + self.env.step/2, fill = 'red')
            self.env.canvas.pack()
            self.env.window.update_idletasks()
            self.env.window.update()
            #time.sleep(0.05)
            while self.P.isover==False and self.P.movements < self.env.max_plays:
                self.env.canvas.delete(o)
                self.P.play()
                o = self.env.canvas.create_oval(self.P.state.y*self.env.step -10 + self.env.step/2, (self.P.state.x)*self.env.step-10 + self.env.step/2, self.P.state.y*self.env.step+10 + self.env.step/2, (self.P.state.x)*self.env.step+10 + self.env.step/2, fill = 'red')
                self.env.canvas.pack()
                self.env.window.update_idletasks()
                self.env.window.update()
                #time.sleep(0.05)
            self.env.canvas.delete(o)
                
            print('score : ' + str(self.P.score))
            #time.sleep(0.5)
            
            #on en récupère le deltamax
            deltamax=self.P.deltamax
            
            #on incrémente le compteur de parties jouées
            n+=1
            print(n)

            #joueur suivant !
            self.birth(epsilon = 1-n/N)

        #notre QTable a été entraînée !

    def playagame(self, epsilon = 0.1, learning=False):
        #joue une partie, par défaut optimale
        self.birth(epsilon)
        o = self.env.canvas.create_oval(self.P.state.y*self.env.step -10 + self.env.step/2, (self.P.state.x)*self.env.step-10 + self.env.step/2, self.P.state.y*self.env.step+10 + self.env.step/2, (self.P.state.x)*self.env.step+10 + self.env.step/2, fill = 'red')
        self.env.canvas.pack()
        self.env.window.update_idletasks()
        self.env.window.update()
        time.sleep(0.1)
        while self.P.isover==False and self.P.movements < self.env.max_plays:
            self.env.canvas.delete(o)
            self.P.play(learning)
            o = self.env.canvas.create_oval(self.P.state.y*self.env.step -10 + self.env.step/2, (self.P.state.x)*self.env.step-10 + self.env.step/2, self.P.state.y*self.env.step+10 + self.env.step/2, (self.P.state.x)*self.env.step+10 + self.env.step/2, fill = 'red')
            self.env.canvas.pack()
            self.env.window.update_idletasks()
            self.env.window.update()
            time.sleep(0.1)
        self.env.canvas.delete(o)

    @staticmethod    
    def nombre(action):
        if action=='Left':
            return 0
        if action == 'Right':
            return 1
        if action == 'Up':
            return 2
        if action == 'Down':
            return 3
        
    def update_QTable(self, previous_state, action, state, reward):
        self.QTable[previous_state.x][previous_state.y][self.nombre(action)] = (1-self.a)*self.QTable[previous_state.x][previous_state.y][self.nombre(action)] + self.a *(reward + self.g * max(self.QTable[state.x][state.y]))

        

    
