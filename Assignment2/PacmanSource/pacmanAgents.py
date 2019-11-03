# Name: Virendra Singh Rajpurohit
# NetID: vsr266
# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
from heuristics import *
import random
import math

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0, 5):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        global limitReached 
        limitReached = False
        possibleActions = state.getAllPossibleActions()
        rootState = state
        for i in range(0,len(self.actionList)):
            self.actionList[i] = random.choice(possibleActions)
        
        tempState = self.genS(state, self.actionList)
        firstSequenceScore = gameEvaluation(rootState,tempState)
        
        while limitReached is False:
            tempSequence = list(self.actionList)
            
            for i in range(0, len(tempSequence)):
                #randomization with 50% 
                if random.randint(0,1) > 0.5:
                    tempSequence[i] = random.choice(possibleActions)
            seqState = self.genS(state, tempSequence)
            
            if seqState == state: seqScore = 0
            else: seqScore = gameEvaluation(rootState,seqState)
            
            #updating actionList if the new sequence has better score
            if seqScore > firstSequenceScore:
                firstSequenceScore = seqScore
                self.actionList = list(tempSequence)
                
        return self.actionList[0]
        
    def genS(self, state, actionSequence):
        #generating state from sequence
        for i in range(0, len(actionSequence)):
            if state.isWin() is True or state.isLose() is True: return state
            seqState = state
            state = state.generatePacmanSuccessor(actionSequence[i])
            if state == None:
                global limitReached
                limitReached = True
                state = seqState
                break
        return state

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0, 5):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        limitReached = False
        popWScore = []
        bestScoreAction = (0,'')
        individual = list(self.actionList)
        possibleActions = state.getAllPossibleActions()
        rootState = state
        #populatoin length is 8
        for i in range(0, 8):
            for l in range(0,len(individual)):
                individual[l] = random.choice(possibleActions)
            #generating 8 individuals for population with random actions from possible
            # and initializing every individual with the score 0
            popWScore.append([list(individual),0])
            
        while limitReached is False :
            for i in range(0,len(popWScore)):
                tState = state
                for l in range(0,len(individual)):
                    if tState.isWin() is True or tState.isLose() is True: break
                    previous = tState
                    tState = tState.generatePacmanSuccessor(popWScore[i][0][l])
                    if tState == None:
                        limitReached = True
                        tState = previous
                        break
                #when limit is reached
                if tState == state:
                    seqScore = 0
                #otherwise
                else:
                    seqScore = gameEvaluation(rootState,tState)
                popWScore[i][1] = seqScore
                
            popWScore = sorted(popWScore, key = lambda score: score[1])
            #getting best individual from best score
            bestIndWScore = (popWScore[-1][1], popWScore[-1][0][0])
            if (bestIndWScore[0] > bestScoreAction[0]):
                bestScoreAction = tuple(bestIndWScore)
            
            modPopulation = []
            #proportionally decreasing probability with rank
            pos = [7]*8 + [6]*7 + [5]*6 + [4]*5 + [3]*4 + [2]*3 + [1]*2 + [0]*1
            
            for i in range (0,4):
                parents=[]
                for x in range(2):
                    #picking two random parents with biased probability
                    parents.append(popWScore[random.choice(pos)][0])
                #crossover
                if (random.randint(0,10) <= 7):
                    os1=[]
                    os2=[]
                    for b in range(0,len(parents[0])):
                        if (random.randint(0,1) < 0.5):
                            os1.append(parents[0][b])
                            os2.append(parents[1][b])
                        else:
                            os1.append(parents[1][b])
                            os2.append(parents[0][b])
                    modPopulation.append(list(os1))
                    modPopulation.append(list(os2))
                else:
                    modPopulation.append(list(parents[0]))
                    modPopulation.append(list(parents[1]))
            #mutation
            for i in range(0,len(modPopulation)):
                if (random.randint(0,10) <=1):
                    mutationPoint = random.randint(0,len(modPopulation[i])-1)
                    modPopulation[i][mutationPoint] = random.choice(possibleActions)
            #updating population
            for i in range(0,len(modPopulation)):
                popWScore[i] = list([modPopulation[i],0])
        return bestScoreAction[1]

class MCTSNode():
    
    def __init__(self,action, parent):
        self.visitCount = 1
        self.children = []
        self.value = 0
        self.parent = parent
        self.action = action
        self.allExpanded = False
        
class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
#        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        global limitReached 
        limitReached = False
        node = MCTSNode(None, None)
        mostVisitAction = []
        rootState = state
        
        while limitReached is False:
            newNode = self.treePolicy(node, state)
            
            if newNode == None:
                # limit reached
                if limitReached is True:break
                # goal reached
                else:continue
            #getting the state for the newNode
            newState = self.getStateFromNode(newNode, state)
            
            if newState == None:
                if limitReached is True:break
                else:continue
                
            value = self.rollout(rootState,newState)
            self.backP(newNode, value)
        
        #getting the best action/s with highest visitCount
        for i in node.children:
            mostVisitAction.append([i.visitCount, i.action])
        mostVisitAction = sorted(mostVisitAction, key=lambda visit: visit[0])
        #sorting and getting the maximum visitCount actoin
        return random.choice([a[1] for a in mostVisitAction if a[0] == mostVisitAction[-1][0]])
    
    def treePolicy(self, node, state):
        unExpanded = False
        while unExpanded is False:
            if node.allExpanded is True:
                node = self.select(node)        
            else:
                unExpanded = True 
        #exit loop only when un-expanded node found
        newNode = self.expand(node, state)
        return newNode
    
    def select(self, node):
        #selecting the node with best value
        bestValueAction = [0,[]]
        for i in node.children:
            result = (i.value/i.visitCount) + math.sqrt((2*math.log(node.visitCount))/i.visitCount)
            if (result == bestValueAction[0]):
                bestValueAction[1].append(i)
            if (result > bestValueAction[0]):
                bestValueAction = [result, [i]]
        return random.choice(node.children) if not bestValueAction[1] else random.choice(bestValueAction[1])
    
    def getStateFromNode(self, node, state):
        #generating state from a node; 
        # returns none when goal, 
        # returns none with limitReached=True when limit reached
        tNode = node
        tState = state
        link = []
        while tNode.parent is not None:
            link.append(tNode)
            tNode = tNode.parent
                    
        link = list(reversed(link))
        
        for i in link:
            previous = tState
            tState = tState.generatePacmanSuccessor(i.action)
            if tState is None:
                global limitReached
                limitReached = True
                self.backP(i, gameEvaluation(state, previous))
                return None
            if tState.isWin() is True or tState.isLose() is True:
                self.backP(i, gameEvaluation(state, tState))
                return None
        return tState
    
    def expand(self, node, state):
        counter = 0
        nextState = self.getStateFromNode(node, state)
        if nextState == None:
            return None
                
        legalActions = nextState.getLegalPacmanActions()
        childActions = [i.action for i in node.children]
        
        for i in range(0, len(legalActions)):
            if legalActions[i] in childActions:
                counter=counter+1
            else:
                #adding new child node with action not present already in parent
                newChildNode = MCTSNode(legalActions[i], node)
                node.children.append(newChildNode)
                break
        
        if counter == len(legalActions):
            node.allExpanded = True
            
        return node.children[-1]
    
    def backP(self, node, value):
        #updating value and visitCount for nodes
        while node is not None:
            node.visitCount = node.visitCount +1
            node.value = node.value + value
            node = node.parent
        return
    
    def rollout(self, rootState, state):
        #rollout limit to 5 depth
        for i in range(0,5):
            previous = state
            if state.isWin() is True or state.isLose() is True : break
            legalActions = state.getLegalPacmanActions()
            state = state.generatePacmanSuccessor(random.choice(legalActions))
            if state is None:
                global limitReached
                limitReached = True
                state = previous
                break
        return gameEvaluation(rootState, state)
