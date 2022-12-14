from random import randint
import math as m
import random
import numpy as np


class StickGame(object):

    def __init__(self, nb, player1, player2, V):
        self.original_nb = nb
        self.nb = nb
        self.V = V
        self.history = []
        self.player1 = player1
        self.player2 = player2



    def play_action(self):
        action1 = self.player1.action(self.nb)
        nb_before_action = self.nb
        int_nb = self.nb - action1
        if int_nb <= 0:
            print('Player2 wins')
            self.history.append((nb_before_action, 0, 0))
            return 0
        if self.player2.is_human:
            self.display(int_nb)
        action2 = self.player2.action(int_nb)
        new_nb = int_nb - action2
        self.history.append((nb_before_action, int_nb, new_nb))
        if new_nb <= 0:
            print('Player1 wins')
            return 0
        else:
            return new_nb

    def display(self, state):
        # Display the state of the game
        print ("| " * state)


    def update_V(self):
        for transition in reversed(self.history):
            (s, si, sp) = transition
            if si == 0:
                self.V[s] = self.V[s] + 0.1*(-1 - self.V[s])
            elif sp == 0:
                self.V[s] = self.V[s] + 0.1*(1 - self.V[s])
            else:
                self.V[s] = self.V[s] + 0.1*(self.V[sp] - self.V[s])


    def game(self):
        while self.nb >0:
            self.display(self.nb)
            self.nb = self.play_action()
        self.update_V()
        self.nb = self.original_nb
        self.history = []

    def run(self):
        games_nb = 500
        e = self.player1.epsilon / games_nb
        for k in range(games_nb):
            print('Partie n° ', k+1)
            self.game()
            self.player1.epsilon += -e
            self.player2.epsilon += -e

        print('Learning done! Lets play!')
        self.player1.is_human = True
        self.game()


class Player(object):

    def __init__(self, is_human):
        self.epsilon = 0.99
        self.is_human = is_human

    def greedy_action(self, state):
        actions = [1, 2, 3]
        vmin = m.inf
        action_min = 1
        for i in actions:
            if state-i > 0 and StickGame.V[state-i] < vmin:
                vmin = StickGame.V[state-i]
                action_min = i
        return action_min

    def random_action(self, state):
        if state == 1:
            return 1
        elif state <= 3:
            return randint(1, state-1)
        else:
            return randint(1,3)

    def human_action(self):
        action = int(input('>:'))
        if action not in [1, 2, 3]:
            action = int(input('>:'))
        return action

    def action(self, state):

        if self.is_human:
            return self.human_action()

        elif random.uniform(0, 1) < self.epsilon:
            return self.random_action(state)

        else:
            return self.greedy_action(state)


if __name__ == '__main__':

    nb = 12
    Human_Player = Player(True)
    IA_Player = Player(False)
    IA_Player2 = Player(False)

    V = {}
    for i in range(nb):
        V[i+1] = 0
    #V = {1: -0.8905810108684877, 2: 0.9202335569231275, 3: 0.9258535339972991, 4: 0.9214785091131977, 5: -0.21781956506726774, 6: 0.7092696350441277, 7: 0.7157271885188823, 8: 0.7272239552828645, 9: -0.04478612673219669, 10: -0.011806034758301547, 11: 0, 12: 0.6448396075359695}


    StickGame = StickGame(nb, IA_Player, IA_Player2, V)
    StickGame.run()


