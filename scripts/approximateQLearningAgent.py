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
    def scale_0_1(self, value, min, max):
        return (value - min) / (max - min)
    
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

        self.weights = np.zeros(len(self.features))
        # Bootstrapping weights
        self.weights[0] = 0 #min trains left
        self.weights[1] = -2 #train urgency
        self.weights[2] = -50 #remaining destination card points
        self.weights[3] = -20 #train cards needed for destination cards
        self.weights[4] = 0 #faceup cards for destination cards
        self.weights[5] = 30 #points gained from last move
        self.weights[6] = 0 #wild cards available
        self.weights[7] = -5 #bias
        
        self.discount = 0.995
        self.alpha = 0.001
        self.weight_decay = 0.9999
        self.weight_decay_final = 0.99
        self.epsilon = 1
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
        self.last_chosen_agent = None
        self.run_failure = False
        self.best_agents_reporting = []
        self.prev_points = 0
    
    def decide(self, game, pnum):        
        #get possible actions from agents
        overall_possible_actions = game.get_possible_moves(pnum)
        try:
            assert len(overall_possible_actions) > 0
        except AssertionError as e:
            print("AQL: no moves left to make")
            self.run_failure = True
            return
        
        possible_actions = []
        for a in self.agents:
            possible_actions.append((a, a.decide(game, pnum)))
        best_val = None
        best_action = None
        best_agent = None
        self.best_agents_reporting = []
        #print("Deciding move")
        for (ag, act) in possible_actions:
            try:
                assert act is not None
            except AssertionError as e:
                print(f"{ag.__class__.__name__} returned a None action")
                self.run_failure = True
                return
            #based on action a
            game_after_a = game.copy()
            game_after_a.make_move(act.function, act.args)
            #compute value of features given game state after taking action a from current game state
            evaluated_features = self.get_features(game_after_a, pnum, ag)
            q_val = np.dot(evaluated_features, self.weights)
            #print("Agent", ag.__class__.__name__, "suggested", act.function, act.args, "with score", q_val)
            if best_val is None or q_val > best_val:
                best_val = q_val
                best_action = act
                best_agent = ag
        
        for (ag, act) in possible_actions:
            if act == best_action:
                self.best_agents_reporting.append(ag)

        self.last_chosen_agent = best_agent
        return best_action
    
    def choose_destination_cards(self, moves, game, pnum, num_keep):
        #get possible actions from agents
        possible_actions = []
        for a in self.agents:
            m_copy = []
            for m in moves:
                m_copy.append(m.copy())
            possible_actions.append((a, a.choose_destination_cards(m_copy, game.copy(), pnum, num_keep)))
        best_val = None
        best_action = None
        best_agent = None
        for (ag, act) in possible_actions:
            assert act is not None, f"{ag.__class__.__name__} returned a None action"
            assert act.function == 'chooseDestinationCards'
            #based on action a
            game_after_a = game.copy()
            game_after_a.choose_destination_cards(pnum, act.args[1], num_keep)
            
            #compute value of features given game state after taking action a from current game state
            evaluated_features = self.get_features(game_after_a, pnum, ag)
            q_val = np.dot(evaluated_features, self.weights)
            if best_val is None or q_val > best_val:
                best_val = q_val
                best_action = act
                best_agent = ag
        
        self.last_chosen_agent = best_agent
        return best_action
    
    def update_final(self, pnum, game_state, reward):
        cur_features = self.get_features(game_state, pnum, None)
        difference = reward - np.dot(cur_features, self.weights)
        self.weights = self.weights * self.weight_decay_final + self.alpha * difference * cur_features
    
    def update(self, pnum, chosen_agent, game_after_action, game_before_next_turn, reward):
        #ISSUE: others will play between now and then!
        #SOLUTION: evaluate right before action based on previous action
        possible_nextgame_actions = []
        for a in self.agents:
            try:
                assert len(game_before_next_turn.get_possible_moves(pnum)) > 0
            except AssertionError as e:
                print(f"face up {game_before_next_turn.train_cards_face_up}")
                print(f"train deck {sum(game_before_next_turn.train_deck.deck.values())}")
                print(f"destination deck {sum(game_before_next_turn.destination_deck.deck.values())}")
                print("hands")
                for (i, p) in enumerate(game_before_next_turn.players):
                    print("Player", i)
                    print(p.hand)
                    print(p.hand_destination_cards)
                self.run_failure = True
                return
                
            possible_nextgame_actions.append((a, a.decide(game_before_next_turn, pnum)))
        
        
        best_future_q = None
        for (ag, act) in possible_nextgame_actions:
            assert act is not None, f"{ag.__class__.__name__} returned a None action"
            game_after_nextgame_a = game_before_next_turn.copy()
            game_after_nextgame_a.make_move(act.function, act.args)
            future_q = np.dot(self.get_features(game_after_nextgame_a, pnum, ag), self.weights)
            if best_future_q is None or best_future_q < future_q:
                best_future_q = future_q
        
        cur_features = self.get_features(game_after_action, pnum, chosen_agent)
        difference = reward + self.discount * best_future_q - np.dot(cur_features, self.weights)
        self.weights = self.weights * self.weight_decay + self.alpha * difference * cur_features
        
        #manual clipping for weights
        self.weights[1] = max(-5, self.weights[1]) #train urgency should never plummet
        self.weights[2] = min(-10, self.weights[2]) #incomplete points should never skyrocket
    
    def weight_function(self, zero_edges):
        def fn(u, v, d):
            if (u, v) in zero_edges or (v, u) in zero_edges:
                return 0
            return d.get("weight", 1)
        return fn
    
    def get_features(self, game, pnum, agent):
        #update joint graph/remaining destinations storage
        self.jgraph = self.joint_graph(game, pnum)
        self.remaining_dest = self.destinations_not_completed(game, pnum, self.jgraph)
        player_graph = game.player_graph(pnum)

        #update paths planned with shortest available path to each remaining dcard
        claimed_edges = set(player_graph.edges())

        #figure out what color cards are needed to complete each of these paths
        #note: the two below variables are different since routes with multiple colors to claim show up 2x in self.cards_needed
        self.cards_needed = emptyCardDict()
        self.num_cards_needed = 0
        for d in self.remaining_dest:
            #this shortest path will include this player's already-claimed routes!
            paths_lst = nx.shortest_path(self.jgraph, d['city1'], d['city2'], weight=self.weight_function(claimed_edges))
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

        res = np.zeros(len(self.features))
        for (i, f) in enumerate(self.features):
            this_res, min, max = f(game, pnum)
            res[i] = self.scale_0_1(this_res, min, max)
            #this_res, min, max = f(game, pnum, agent)
            #res.append(self.scale_0_1(this_res, min, max))
        return res
    
    #def feature_num_players(self, game, pnum): #2-5
        #return game.number_of_players, 2, 5
    
    def feature_minimum_trains_left(self, game, pnum): #2-45
        return min([p.number_of_trains for p in game.players]), 2, 45
    
    def feature_train_urgency(self, game, pnum): #0-1
        return 1 - game.players[pnum].number_of_trains / 45, 0, 1
    
    def feature_dcard_points_remaining(self, game, pnum): #vaguely 0-63 (3 tickets @ 22+21+20)
        res = 0
        for dcard in self.remaining_dest:
            res += dcard['points']
        return res, 0, 63
    
    def feature_num_traincar_cards_for_dcards_remaining(self, game, pnum): #vaguely 0-63 (cards needed same as dcard points)
        return self.num_cards_needed, 0, 63

    def feature_faceup_cards_for_dcards_remaining(self, game, pnum): #0-5
        count = 0
        cards_face_up = {k: v for k, v in game.train_cards_face_up.items() if v != 0}
        
        for c in cards_face_up:
            if self.cards_needed[c] > 0:
                count += min(self.cards_needed[c], cards_face_up[c])
        
        return count, 0, 5
    
    def feature_points_earned(self, game, pnum): #0-15
        delta = game.players[pnum].points - self.prev_points
        self.prev_points = game.players[pnum].points
        return delta, 0, 15

    def feature_wild_cards_available(self, game, pnum): #0-5
        return game.train_cards_face_up["wild"], 0, 5
    
    def feature_bias(self, game, pnum):
        return 1, 0, 1
    
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
            return self.agents[rand_agent_idx].decide(game.copy(), pnum), self.agents[rand_agent_idx]
        else:
            move = self.decide(game, pnum) #will copy game
            return move, self.last_chosen_agent
    
    def train_choose_destination_cards(self, moves, game, pnum, num_keep):
        #with probability epsilon, randomly decide between the agents
        #with probabiliity 1-epsilon, just take the real action
        if random.random() < self.epsilon:
            rand_agent_idx = random.randint(0, len(self.agents) - 1)
            return self.agents[rand_agent_idx].choose_destination_cards(moves.copy(), game.copy(), pnum, num_keep), self.agents[rand_agent_idx]
        else:
            move = self.choose_destination_cards(moves, game, pnum, num_keep) #will copy moves/game
            return move, self.last_chosen_agent