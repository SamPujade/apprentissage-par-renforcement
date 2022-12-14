from numpy import math #on importe le module math pour l'infini et sqrt
#from matplotlib.pyplot import *
from random import choice, random
from tkinter import *
import time
from PIL import ImageGrab


class environment():
    """Une instance de cette classe est un environnement, un terrain de jeu, tout ce qui enveloppe le reste."""

    def __init__(self, n, rewarded_states=[], rewards=[], terminal_states=[], forbidden_states=[], start_point=[0, 0], max_plays = 30, move_penalty = -1):

        self.states=[[] for k in range(n)] #la matrice des états de l'environnement = cases.
        #ATTENTION: pour suivre les coordonées Tkinter, c'est une liste de colonnes descendantes. x:gauche->droite ; y:haut->bas

        self.max_plays = max_plays #le nombre de coups (actions) maximum autorisés par partie
        self.move_penalty = move_penalty #
        self.len=n #pratique pour aprèsla récompense fixe attribuée pour chaque action

        #le choix de créer une liste d'états récompensés et une liste des récompenses
        #vient du fait qu'en général il y a peu d'état récompensés et donc
        #il serait lourd d'avoir juste une grosse matrice de récompense quasi-nulle


        #création des états (objets)
        for x in range(n):
            for y in range(n):
                reward=0
                is_terminal=False
                if [x, y] in rewarded_states:
                    indice=rewarded_states.index([x, y])
                    reward=rewards[indice]
                if [x, y] in terminal_states:
                    is_terminal=True
                self.states[x]+=[state(x, y, reward, is_terminal)]

        #rensignement de l'état de départ de l'environnement
        self.start = self.states[start_point[0]][start_point[1]] #état de départ

        #éliminations des actions possibles depuis chaque état selon position (bord) ou états interdits adjacents
        for x in range(n):
            for y in range(n):
                if x==0 or [x-1, y] in forbidden_states:
                    self.states[x][y].possible_actions.remove('Left')
                elif x==n-1 or [x+1, y] in forbidden_states:
                    self.states[x][y].possible_actions.remove('Right')
                if y==0 or [x, y-1] in forbidden_states:
                    self.states[x][y].possible_actions.remove('Up')
                elif y==n-1 or [x, y+1] in forbidden_states:
                    self.states[x][y].possible_actions.remove('Down')

        #initialisation de l'interface graphique
        def color(self, s): #s c'est un state
            return 'red' if s.reward<0 else 'yellow' if s.reward>0 else 'black' if [s.x, s.y] in forbidden_states else 'blue' if s == self.start else None

        #on a a besoin pour après, pas seulement en local
        self.size = int(600/n)*n
        self.step = self.size / n

        self.root = Tk()
        self.root.title('Jeu 2D interactif')

        self.canvas = Canvas(self.root, height=self.size+1+13, width=self.size+1) #le +1 c'est pour les lignes droite et bas
        #self.canvas.create_text(1, self.size+1,anchor='nw', text='Nombre maximal de coups : {} ; Penalité de mouvement : {}'.format(self.max_plays, self.move_penalty))

        #on représente les états par des carrés adéquats sur une grille
        for i in range(n):
            for j in range(n):
                s=self.states[i][j]
                self.canvas.create_rectangle(2+i*self.step, 2+j*self.step, 2+(i+1)*self.step, 2+(j+1)*self.step, fill = color(self, s), width=1, tag = '%d_%d'%(i,j))
                if s.reward!=0:
                    self.canvas.create_text(2+(i+0.5)*self.step, 2+(j+0.5)*self.step, text='{}'.format(s.reward), anchor='center')

        self.canvas.pack(side='left')

        #dans un canvas tkinter, les coordonnées dessinables vont de 2 à n+1 où n est une dimension

#cette classe est aussi utile qu'un dictionnaire qui aurait les mêmes attributs, mais c'est sympa comme ça
class state():
    """Une instance de cette classe est un état de l'environnement, elle est donc générée automatiquement par un environnement
     Un objet état contient toutes les informations nécessaire relatives à lui-même : position sur la grille,
    actions possibles, récompense, terminalité."""

    def __init__(self, x, y, reward, is_terminal):
        self.x=x
        self.y=y
        self.reward=reward
        self.is_terminal=is_terminal
        self.possible_actions=['Left', 'Right', 'Up', 'Down']


#Un petit environnement sympathique
#E=environment(n=7, rewarded_states=[[0,0],[1,0],[6,0],[1,1],[5,1],[4,2],[4,6],[5,6]],
                #terminal_states=[[1,0],[6,0],[1,1],[4,2]], rewards=[100, -1000, -1000, -1000, 10, -1000, 2, 2], start_point=[1, 5])

E=environment(n=4, rewarded_states=[[2, 0], [1, 1]], terminal_states=[[2, 0], [1, 1]], rewards=[100, -100], start_point=[0, 3])




class game():
    """Une instance de cette classe simule une partie du jeu.
Dépend donc d'un objet environnement. Et d'un objet joueur capable de jouer beaucoup de parties.
Dispose de méthode permettant de faire avancer la partie en jouant selon diverses stratégies possibles."""

    def __init__(self, environnement, player, epsilon, tau):
        self.state = environnement.start #on commence la partie sur l'état de départ de l'environnement
        self.previous_state=None
        self.score = 0 #le score cumulatif qui va être réalisé sur la partie
        self.is_over = False #est-ce que la partie est finie
        self.deltamax = 0 #la valeur abs de la correction maximale effectuée sur la QTable du superobjet joueur pendant la partie

        self.P = player
        self.e = epsilon
        self.tau = tau
        self.env = environnement

        self.movements = 0 #le nombre de coups joués sur cette partie. Eh oui, l'environnement limite le nombre de coups jouables par partie.


    def play(self, learning=True, method='e_greedy'): #epsilon-greedy method or softmax or optimal (just play one of the best actions)
        """Effectue une action en actualisant la QTable du joueur et le deltamax ou pas (selon la valeur de learning), et en actualisant le nombre de mouvements, le score et la terminalité."""

        if method=='e_greedy': #la méthode epsilon-greedy
            if random()<self.e:
                order = choice(self.state.possible_actions)
            else:
                order=choice(self.best_actions())

        elif method=='softmax': #la méthode softmax
            QTotal = sum([exp(self.P.QTable[self.state.x][self.state.y][action]/self.tau) for action in self.state.possible_actions])
            if QTotal == 0:
                order = choice(self.state.possible_actions)
            else:
                Proba={} #dico de probas
                for action in self.state.possible_actions:
                    Proba[action]=exp( self.P.QTable[self.state.x][self.state.y][action] / self.tau ) / QTotal
                alea=random() #entre 0 et 1
                if alea<Proba['Left']:
                    order = 'Left'
                elif alea<Proba['Left']+Proba['Right']:
                    order = 'Right'
                elif alea<Proba['Left']+Proba['Right']+Proba['Up']:
                    order = 'Up'
                else:
                    order = 'Down'
                #pas de risque d'action impossible

        elif method=='optimal': #action optimale
            order=choice(self.best_actions())


        #maintenant on gère ce qu'il y a à gérer pour exécuter l'ordre de jeu donné par la méthode choisie
        self.previous_state = self.state
        self.move(order)
        self.movements += 1
        self.score+=self.state.reward + self.env.move_penalty

        if learning: #update de la QTable si learning==True
            old_value = self.P.QTable[self.previous_state.x][self.previous_state.y][order]
            self.P.update_QTable(self.previous_state, order, self.state, self.state.reward + self.env.move_penalty)
            correction = abs(old_value - self.P.QTable[self.previous_state.x][self.previous_state.y][order])
            if correction > self.deltamax :
                self.deltamax = correction

        if self.state.is_terminal == True or self.movements==self.env.max_plays: #on vérifie qu'on a pas fini la partie
            self.is_over = True


    def best_actions(self):
        liste=[]
        Q=self.P.QTable[self.state.x][self.state.y]
        for action in self.state.possible_actions:
            if Q[action] == max(Q.values()): #tout fonctionne parce que les seules clés de Q sont des actions possibles depuis l'état
                liste+=[action]
        return liste


    def move(self, order):
        if order=='Left':
            self.state=self.env.states[self.state.x-1][self.state.y]
        elif order=='Right':
            self.state=self.env.states[self.state.x+1][self.state.y]
        elif order=='Up':
            self.state=self.env.states[self.state.x][self.state.y-1]
        elif order=='Down':
            self.state=self.env.states[self.state.x][self.state.y+1]



class player():
    """Un objet de cette classe va être un superobjet qui contiendra des informations
sur un entraînement réalisé avec une série de parties. Il représente donc un joueur virtuel qui apprend en jouant des parties successives.
La méthode train() va instancier
des objets game (= va créer des parties) et les faire jouer jusqu'à un
état terminal (=va jouer les parties jusqu'au bout) jusqu'à atteindre les critères de convergence.
De là, on pourra accéder à la QTable obtenue suite à cet entraînement, et faire
jouer des parties sans la modifier pour observer le comportement en régime établi,
ou reprendre l'entraînement.
C'est un tel objet qui va être repésenté graphiquement, car il dépend d'un environnement
qu'on pourra représenter, et contient des games qui pourront être représentés
successivement, ainsi qu'une QTable évolutive."""

    def __init__(self, env = E, alpha = 0.75, gamma = 0.9):
        self.env = env #le terrain de jeu

        #la partie
        self.G = game(self.env, self, 1, 20)#juste pour ne pas qu'il y ait d'erreur ; cette partie ne sera jamais jouée et sera écrasée

        #la QTable du joueur , prenant en paramètre les coordonnées puis l'action
        self.QTable=[[{action:0 for action in env.states[x][y].possible_actions} for y in range(self.env.len)]for x in range(self.env.len)]
        #grosse pythonerie, prendre le temps de lire. Et on initialise des Q valeurs que pour les actions possibles à chaque état

        #notre joueur doit avoir un QTable limite unique (dépend de l'environnement et des paramètres alpha et gamma) vers laquelle on tend en l'entraînant
        self.a = alpha
        self.g = gamma

        self.train_nb = 0 #nombre de parties jouées

        self.tempo=True #pour savoir si on affiche les parties lentement pour voir les actions ou si on les joue très vite

        try:
            self.env.canvas.delete('arrow')#enlever les éventuelles flèches d'un précédent joueur
        except:
            pass


    def new_game(self, epsilon, tau):
        self.G = game(self.env, self, epsilon, tau)


    def train(self, N = 200, stop = True , threshold = 0.001, train_type = 'inverseroot', method='e_greedy', epsilon=0.1, tau=20): #on train sur plusieurs parties en explorant d'une certaine façon
        """Fait jouer des joueurs successifs et arrête l'entraînement soit quand le deltamax de la dernière partie jouée est inférieur au seuil (si stop est true)
    soit après avoir fait jouer N joueurs. La méthode renseigne comment vont être jouées les parties, et le type d'entraînement permet un raffinement de la
    manière dont vont être générée les taux d'exploration des parties si on utilise epsilon-greedy. Par exemple, on peut sélectionner une décroissance linéaire."""

        self.new_game(1, tau)#le premier joueur est entièrement aléatoire si on utilise la méthode epsilon-greedy, et est normal avec softmax
        deltamax = math.inf

        while (stop and deltamax>threshold and self.train_nb<N) or (stop==False and self.train_nb<N):
            #on joue des parties

            #position du joueur sur le canvas
            self.graphic_update()

            #pour éventuellement jouer les parties lentement
            if self.tempo:time.sleep(0.05)

            while self.G.is_over==False:
                #on joue un coup
                self.G.play(method=method)

                #actualisation des informations visuelles
                #self.show_arrows()
                self.graphic_update()

                if self.tempo:time.sleep(0.05)

            if self.tempo:time.sleep(0.5)

            #on en récupère le deltamax
            deltamax=self.G.deltamax

            #on incrémente le compteur de parties jouées
            self.train_nb+=1

            #nouvelle partie !
            if train_type=='linear':
                self.new_game(1-self.train_nb/N, tau)
            elif train_type=='constant':
                self.new_game(epsilon, tau)
            elif train_type=='exponential':
                self.new_game(math.exp(-self.train_nb), tau)
            elif train_type=='quadratic':
                self.new_game((1-self.train_nb/N)**2, tau)
            elif train_type=='inverseroot':
                self.new_game(1/math.sqrt(self.train_nb + 1), tau)

        #notre QTable a été entraînée !
        self.env.canvas.delete('playerpos')
        print(self.train_nb)



    def switch_tempo(self): #methode pour interaction via IHM
        self.tempo = not self.tempo


    def play(self, epsilon, learning=False, method='e_greedy', tau = 20):
        #joue une partie, par défaut optimale, sans update la QTable
        self.new_game(epsilon, tau)#balec si learning==False
        self.graphic_update()
        ImageGrab.grab((0, 220, 608, 608)).save("IMAGE0.png")
        time.sleep(0.1)
        while self.G.is_over==False:
            self.G.play(learning, method)
            self.graphic_update()
            time.sleep(0.1)
        self.env.canvas.delete('playerpos')

        return self.G.score


    def update_QTable(self, previous_state, action, state, reward):
        self.QTable[previous_state.x][previous_state.y][action] = (1-self.a)*self.QTable[previous_state.x][previous_state.y][action] + self.a *(reward + self.g * max(list(self.QTable[state.x][state.y].values())))

    def graphic_update(self):
        self.env.canvas.delete('playerpos')
        self.env.canvas.create_oval(self.G.state.x*self.env.step -10 + self.env.step/2,
                                            (self.G.state.y)*self.env.step-10 + self.env.step/2,
                                            self.G.state.x*self.env.step+10 + self.env.step/2,
                                            (self.G.state.y)*self.env.step+10 + self.env.step/2,
                                            fill = 'red', tag='playerpos')
        self.env.canvas.pack()
        self.env.root.update_idletasks()
        self.env.root.update()


    def show_QTable(self): #à optimiser comme pour les flèches, là c'est giga lourd
        self.env.canvas.delete('texte')
        for i in range(self.env.len):
            for j in range(self.env.len):
                s=self.env.states[i][j]
                if 'Left' in s.possible_actions:
                    self.env.canvas.create_text(2+i*self.env.step, 2+(j+0.5)*self.env.step, text=str(int(self.QTable[i][j]['Left'])), anchor='w', tag='texte')
                if 'Right' in s.possible_actions:
                    self.env.canvas.create_text(2+(i+1)*self.env.step, 2+(j+0.5)*self.env.step, text=str(int(self.QTable[i][j]['Right'])), anchor='e', tag='texte')
                if 'Up' in s.possible_actions:
                    self.env.canvas.create_text(2+(i+0.5)*self.env.step, 2+j*self.env.step, text=str(int(self.QTable[i][j]['Up'])), anchor='n', tag='texte')
                if 'Down' in s.possible_actions:
                    self.env.canvas.create_text(2+(i+0.5)*self.env.step, 2+(j+1)*self.env.step, text=str(int(self.QTable[i][j]['Down'])), anchor='s', tag='texte')
        return

    def show_arrows(self):
        i, j = self.G.state.x, self.G.state.y
        self.env.canvas.delete('flèche_%d_%d'%(i, j))
        if 'Left' in self.G.best_actions():
            self.env.canvas.create_line(2+(i+0.25)*self.env.step, 2+(j+0.5)*self.env.step,2+(i-0.25)*self.env.step, 2+(j+0.5)*self.env.step, arrow='last', tag=('flèche_%d_%d'%(i, j), 'arrow'))
        if 'Right' in self.G.best_actions():
            self.env.canvas.create_line(2+(i+0.75)*self.env.step, 2+(j+0.5)*self.env.step,2+(i+1.25)*self.env.step, 2+(j+0.5)*self.env.step, arrow='last', tag=('flèche_%d_%d'%(i, j), 'arrow'))
        if 'Up' in self.G.best_actions():
            self.env.canvas.create_line(2+(i+0.5)*self.env.step, 2+(j+0.25)*self.env.step,2+(i+0.5)*self.env.step, 2+(j-0.25)*self.env.step, arrow='last', tag=('flèche_%d_%d'%(i, j), 'arrow'))
        if 'Down' in self.G.best_actions():
            self.env.canvas.create_line(2+(i+0.5)*self.env.step, 2+(j+0.75)*self.env.step,2+(i+0.5)*self.env.step, 2+(j+1.25)*self.env.step, arrow='last', tag=('flèche_%d_%d'%(i, j), 'arrow'))
        return



#faire des stats sur cet environnement, avec les méthodes : eps-greedy fixe (plot nb de game de conv et score)
#                                                           softmax à tau fixe(idem)
#                                                           eps-greedy à décroissance linéaire (sans condition de convergence, pour un nb total de game variable, et score
#                                                           eps-greedy à décroissance quadratique et exponentielle (idem)
#                                                           softmax à décroissance


player1=player()
def new_player():
    global player1
    player1=player()
    return
def train_player1():
    global player1
    player1.train()
    return
def switch_tempo_player1():
    global player1
    player1.switch_tempo()
    return
def play_player1():
    global player1
    player1.play(scale_epsilon.get())
    return



frame = Frame(E.root, height = E.size+14, width = 100)
frame.pack(side='right')

switch_tempo_player1()
train_player1()
switch_tempo_player1()

scale_epsilon = Scale(frame, orient='horizontal', from_=0, to=1, resolution=0.1, label='Régler epsilon:')
scale_epsilon.pack()

button_play=Button(frame, command=play_player1, text='Jouer une partie')
button_play.pack()

from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()


E.root.mainloop()
