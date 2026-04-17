from agent import Agent
import networkx as nx
from ttrengine import emptyCardDict
from hungryAgent import HungryAgent
from longRouteJunkieAgent import LongRouteJunkieAgent
from oneStepThinkerAgent import OneStepThinkerAgent
from pathAgent import PathAgent
import numpy as np
import random

#Question: how to handle storing feature weights?
#   Answer: include training function that only needs to be run once (and stores results with pickle to load in at __init__)

#Glenn: features involving the action - know what action each of the four agents would pick
#       "this action scores points for me, completes card, claims route, picks useful traincard etc."

# Make sure to: train against rotating set of agents
#           2-player game against each alone equal # of times? 4-player game against 2 of 3 rotating?        
#

# 

class ApproximateQLearningAgent(Agent):
    #convert list of True/False values to number corresponding to state
    def bool_list_to_number(self, state):
        state_idx = 0
        for i in range(len(state)):
            if state[i] == True:
                state_idx += 2 ** i
        return state_idx
    
    def point_lookup(self, val):
        tab = {1:1, 2:2, 3:4, 4:7, 5:10, 6:15, 8:21, 9:27}
        return tab[val]
    
    def __init__(self):
        #Just need for other init stuff
        self.agents = [HungryAgent(), LongRouteJunkieAgent(), OneStepThinkerAgent(), PathAgent()]

        #Class variables that should NOT change between training runs

        self.features = [getattr(self, name) for name in self.__class__.__dict__ 
                         if name.startswith("feature_") and callable(getattr(self, name))]
        
        #self.features = []
        #for a in self.agents:
            #for f in feature_lst:
                #self.features.append(lambda game, pnum, given_agent, f=f, a_type=type(a): f(game, pnum) if type(given_agent) == a_type else (0, 0, 1))

        # self.weights = np.zeros(len(self.features))
        # # Bootstrapping weights
        # self.weights[0] = 0 #min trains left
        # self.weights[1] = -2 #train urgency
        # self.weights[2] = -50 #remaining destination card points
        # self.weights[3] = -20 #train cards needed for destination cards
        # self.weights[4] = 0 #faceup cards for destination cards
        # self.weights[5] = 30 #points gained from last move
        # self.weights[6] = 0 #wild cards available
        # self.weights[7] = -5 #bias
        # self.weight_decay = 0.9999
        # self.weight_decay_final = 0.99

        self.qvalues = {state: {agent.__class__.__name__: 10.0 for agent in self.agents} for state in range(2**len(self.features))}
        
        self.discount = 0.995
        self.alpha = 0.15
        self.epsilon = 0.1
        self.reinitialize_vars()        
        
    def reinitialize_vars(self):
        #Class variables that SHOULD change between training runs

        #List of agents to pick actions from.
        #Each agent will have decide() called on it with this agent's pnum, which should keep their fields updated
        self.agents = [HungryAgent(), LongRouteJunkieAgent(), OneStepThinkerAgent(), PathAgent()]
        
        self.jgraph = None
        self.remaining_dest = []
        self.cards_needed = emptyCardDict()
        self.num_cards_needed = 0
        self.run_failure = False
        self.best_agents_reporting = []
        self.last_move_route_points = 0
    
    def decide(self, game, pnum):        
        overall_possible_actions = game.get_possible_moves(pnum)
        try:
            assert len(overall_possible_actions) > 0
        except AssertionError as e:
            print("AQL: no moves left to make")
            self.run_failure = True
            return
        
        #get possible actions from agents no matter what so they update their own state
        agents_chosen_actions = {}
        for a in self.agents:
            action = a.decide(game, pnum)
            try:
                assert action is not None
            except AssertionError as e:
                print(f"{a.__class__.__name__} returned a None action")
                self.run_failure = True
                return
            agents_chosen_actions[a.__class__.__name__] = action

        self.best_agents_reporting = []

        #determine state
        state = self.get_state_from_features(game, pnum)

        #convert list of True/False values to number corresponding to state
        state_idx = self.bool_list_to_number(state)

        #get agent values mapping from table
        agent_to_values_map = self.qvalues[state_idx]

        #find maximum value move
        best_agent_name = None
        best_value = -1 * float('inf')
        for ag_name in agent_to_values_map:
            if agent_to_values_map[ag_name] > best_value:
                best_agent_name = ag_name
                best_value = agent_to_values_map[ag_name]

        best_action = agents_chosen_actions[best_agent_name]
        
        for ag_name in agents_chosen_actions:
            if agents_chosen_actions[ag_name] == best_action:
                self.best_agents_reporting.append(ag_name)

        return best_action, state_idx, best_agent_name
    
    def choose_destination_cards(self, moves, game, pnum, num_keep):
        #get possible actions from agents no matter what so they update their own state
        agents_chosen_actions = {}
        for a in self.agents:
            m_copy = []
            for m in moves:
                m_copy.append(m.copy())
            
            action = a.choose_destination_cards(m_copy, game.copy(), pnum, num_keep)
            
            try:
                assert action is not None
            except AssertionError as e:
                print(f"{a.__class__.__name__} returned a None action")
                self.run_failure = True
                return

            agents_chosen_actions[a.__class__.__name__] = action
        
        #determine state
        state = self.get_state_from_features(game, pnum)

        #convert list of True/False values to number corresponding to state
        state_idx = self.bool_list_to_number(state)

        #get agent values mapping from table
        agent_to_values_map = self.qvalues[state_idx]

        #find maximum value move
        best_agent_name = None
        best_value = -1 * float('inf')
        for ag in agent_to_values_map:
            if agent_to_values_map[ag] > best_value:
                best_agent_name = ag
                best_value = agent_to_values_map[ag]
                
        best_action = agents_chosen_actions[best_agent_name]

        return best_action, state_idx, best_agent_name
    
    def update(self, pnum, state_idx_before_action, chosen_agent_name, game_before_next_turn, reward):
        #ISSUE: others will play between now and then!
        #SOLUTION: evaluate right before action based on previous action

        #game_before_next_turn = None signifies terminal state
        fut_best_value = 0 #if terminal state
        if game_before_next_turn is not None:
            #nonterminal state
            
            #find future state in qvalue table
            fut_state = self.get_state_from_features(game_before_next_turn, pnum)
            fut_state_idx = self.bool_list_to_number(fut_state)
            fut_agent_to_values_map = self.qvalues[fut_state_idx]

            #find best future Q-value
            fut_best_value = -1 * float('inf')
            for ag_name in fut_agent_to_values_map:
                if fut_agent_to_values_map[ag_name] > fut_best_value:
                    fut_best_value = fut_agent_to_values_map[ag_name]
        
        #find original Q-value
        orig_value = self.qvalues[state_idx_before_action][chosen_agent_name]

        #calculate TD difference
        difference = reward + self.discount * fut_best_value

        #update Q-value table
        self.qvalues[state_idx_before_action][chosen_agent_name] = (1 - self.alpha) * orig_value + self.alpha * difference
    
    def get_state_from_features(self, game, pnum):
        #update joint graph/remaining destinations storage
        self.jgraph = self.joint_graph(game, pnum)
        self.remaining_dest = self.destinations_not_completed(game, pnum, self.jgraph)
        player_graph = game.player_graph(pnum)

        #update paths planned with shortest available path to each remaining dcard

        #figure out what color cards are needed to complete each of these paths
        #note: the two below variables are different since routes with multiple colors to claim show up 2x in self.cards_needed
        self.cards_needed = emptyCardDict()
        self.num_cards_needed = 0
        for d in self.remaining_dest:
            #this shortest path will include this player's already-claimed routes!
            paths_lst = nx.shortest_path(self.jgraph, d['city1'], d['city2'], weight='weight')
            for i in range(0, len(paths_lst) - 1):
                node1 = paths_lst[i]
                node2 = paths_lst[i+1]
                #first, check if route already claimed by this player
                if node1 in player_graph.nodes() and node2 in player_graph.nodes():
                    if node1 in player_graph[node2] or node2 in player_graph[node1]:
                        #already claimed - don't need any more cards
                        continue

                #otherwise, find pairing on game board
                edgelist = game.board.get_free_connection(node1, node2, number_of_players=game.number_of_players)
                
                self.num_cards_needed += edgelist[0]['weight']

                #decision: if route has multiple ways to claim it, add to both
                for e in edgelist:
                    #note: graph has colors uppercase but cards_needed has colors lowercase
                    #note: if e['color'] is gray, will just add gray to self_cards_needed. That's OK.
                    col = e['color'].lower()
                    if col == 'gray' and 'gray' not in self.cards_needed:
                        self.cards_needed['gray'] = e['weight']
                    else:
                        self.cards_needed[col] += e['weight']

        res = np.zeros(len(self.features), dtype=bool)
        for (i, f) in enumerate(self.features):
            res[i] = f(game, pnum) #each feature will return True or False
        
        return res
    
    def feature_minimum_trains_left(self, game, pnum): #2-45
        return min([p.number_of_trains for p in game.players]) > ((45-2) / 2)
    
    def feature_train_urgency(self, game, pnum): #0-1
        return game.players[pnum].number_of_trains / 45 > 0.5
    
    def feature_dcard_points_remaining(self, game, pnum): #vaguely 0-63 (3 tickets @ 22+21+20)
        # res = 0
        # for dcard in self.remaining_dest:
        #     res += dcard['points']
        # return res > (63 / 2)
        return len(self.remaining_dest) > 0
    
    def feature_num_traincar_cards_for_dcards_remaining(self, game, pnum): #vaguely 0-63 (cards needed same as dcard points)
        return self.num_cards_needed > (63 / 2)

    def feature_faceup_cards_for_dcards_remaining(self, game, pnum): #0-5
        count = 0
        cards_face_up = {k: v for k, v in game.train_cards_face_up.items() if v != 0}
        
        for c in cards_face_up:
            if self.cards_needed[c] > 0:
                count += min(self.cards_needed[c], cards_face_up[c])
        
        return count > 2
    
    def feature_points_earned(self, game, pnum): #0-15
        return self.last_move_route_points > 3.5

    def feature_wild_cards_available(self, game, pnum): #0-5
        return game.train_cards_face_up["wild"] > 0
    
    #Ideas for other features: Relative score (agent score - max opponent score), tickets completed / total tickets, 45 - min(opp trains left)

    def train_decide(self, game, pnum):
        #with probability epsilon, randomly decide between the agents
        #with probabiliity 1-epsilon, just take the real action
        try:
            assert len(game.get_possible_moves(pnum)) > 0
        except AssertionError as e:
            print(f"face up {game.train_cards_face_up}")
            print(f"train deck {sum(game.train_deck.deck.values())}")
            print(f"destination deck {sum(game.destination_deck.deck.values())}")
            exit(1)
            
        if random.random() < self.epsilon:
            rand_agent_idx = random.randint(0, len(self.agents) - 1)
            move = self.agents[rand_agent_idx].decide(game.copy(), pnum)
            try:
                assert move is not None
            except AssertionError as e:
                print(f"{self.agents[rand_agent_idx].__class__.__name__} returned a None action")
                self.run_failure = True
                return
            
            #determine state, convert list of True/False values to number corresponding to state
            state = self.get_state_from_features(game, pnum)
            state_idx = self.bool_list_to_number(state)

            if move.function == "claimRoute":
                conn = game.board.get_connection(move.args[0], move.args[1])
                self.last_move_route_points = self.point_lookup(conn[0]['weight'])
            else:
                self.last_move_route_points = 0
            return move, state_idx, self.agents[rand_agent_idx].__class__.__name__
        else:
            move, idx, name =  self.decide(game, pnum) #will copy game
            if move.function == "claimRoute":
                conn = game.board.get_connection(move.args[0], move.args[1])
                self.last_move_route_points = self.point_lookup(conn[0]['weight'])
            else:
                self.last_move_route_points = 0
            return move, idx, name
    
    def train_choose_destination_cards(self, moves, game, pnum, num_keep):
        #with probability epsilon, randomly decide between the agents
        #with probabiliity 1-epsilon, just take the real action
        if random.random() < self.epsilon:
            rand_agent_idx = random.randint(0, len(self.agents) - 1)
            move = self.agents[rand_agent_idx].choose_destination_cards(moves.copy(), game.copy(), pnum, num_keep)
            try:
                assert move is not None
            except AssertionError as e:
                print(f"{self.agents[rand_agent_idx].__class__.__name__} returned a None action")
                self.run_failure = True
                return

            #determine state, convert list of True/False values to number corresponding to state
            state = self.get_state_from_features(game, pnum)
            state_idx = self.bool_list_to_number(state)

            return move, state_idx, self.agents[rand_agent_idx].__class__.__name__
        else:
            return self.choose_destination_cards(moves, game, pnum, num_keep) #will copy moves/game