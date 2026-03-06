from agent import Agent
import networkx as nx
from ttrengine import emptyCardDict
from hungryAgent import HungryAgent
from longRouteJunkieAgent import LongRouteJunkieAgent
from oneStepThinkerAgent import OneStepThinkerAgent
from pathAgent import PathAgent
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
    def __init__(self):
        #Class variables that should NOT change between training runs

        self.features = [getattr(self, name) for name in self.__class__.__dict__ 
                         if name.startswith("feature_") and callable(getattr(self, name))]
        
        self.weights = [0] * len(self.features)
        self.discount = 0.995
        self.alpha = 0.0001
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
    
    def decide(self, game, pnum):        
        #get possible actions from agents
        overall_possible_actions = game.get_possible_moves(pnum)
        assert len(overall_possible_actions) > 0, "No moves left to make"
        possible_actions = []
        for a in self.agents:
            possible_actions.append(a.decide(game.copy(), pnum))
        best_val = None
        best_action = None
        for (i, act) in enumerate(possible_actions):
            assert act is not None, f"{self.agents[i].__class__.__name__} returned a None action"
            #based on action a
            game_after_a = game.copy()
            game_after_a.make_move(act.function, act.args)
            #compute value of features given game state after taking action a from current game state
            evaluated_features = self.get_features(game_after_a, pnum)
            #TODO: scale features?
            q_val = sum([f * w for f, w in zip(evaluated_features, self.weights)])
            if best_val is None or q_val > best_val:
                best_val = q_val
                best_action = act
        
        #take action
        return best_action
    
    def choose_destination_cards(self, moves, game, pnum, num_keep):
        #get possible actions from agents
        possible_actions = []
        for a in self.agents:
            m_copy = []
            for m in moves:
                m_copy.append(m.copy())
            possible_actions.append(a.choose_destination_cards(m_copy, game.copy(), pnum, num_keep))
        best_val = None
        best_action = None
        for (i, act) in enumerate(possible_actions):
            assert act is not None, f"{self.agents[i].__class__.__name__} returned a None action"
            assert act.function == 'chooseDestinationCards'
            #based on action a
            game_after_a = game.copy()
            game_after_a.choose_destination_cards(pnum, act.args[1], num_keep)
            
            #compute value of features given game state after taking action a from current game state
            evaluated_features = self.get_features(game_after_a, pnum)
            #TODO: scale features?
            q_val = sum([f * w for f, w in zip(evaluated_features, self.weights)])
            if best_val is None or q_val > best_val:
                best_val = q_val
                best_action = act
        
        return best_action
    
    def weight_function(self, zero_edges):
        def fn(u, v, d):
            if (u, v) in zero_edges or (v, u) in zero_edges:
                return 0
            return d.get("weight", 1)
        return fn
    
    def get_features(self, game, pnum):
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

        return [f(game, pnum) for f in self.features]
    
    def feature_num_players(self, game, pnum):
        return game.number_of_players
    
    def feature_minimum_trains_left(self, game, pnum):
        return min([p.number_of_trains for p in game.players])
    
    def feature_longest_path(self, game, pnum):
        overall_max = 0
        own_max = 0
        for i in range(game.number_of_players):
            player_graph = game.player_graph(i)
            longest_by_node = [game.findMaxWeightSumForNode(player_graph, v, []) for v in player_graph.nodes()]
            this_max = max(longest_by_node) if len(longest_by_node) > 0 else 0
            if i == pnum:
                own_max = this_max
            if this_max > overall_max:
                overall_max = this_max
        
        return overall_max - own_max + 1 #to avoid zeroing out in vector multiplication

    # def feature_minimum_possible_dcard_paths(self, game, pnum):
    #     if len(self.remaining_dest) == 0:
    #         return 0 #zero out in vector multiplication
        
    #     min_poss_paths = float('inf')
    #     for dcard in self.remaining_dest:
    #         #get possible paths to dcard
    #         node1 = dcard['city1']
    #         node2 = dcard['city2']
    #         print("nodes", node1, node2)
    #         print("in jgraph?", node1 in self.joint_graph(game, pnum).nodes(), node2 in self.joint_graph(game, pnum).nodes())
    #         paths_generator = nx.all_simple_paths(self.joint_graph(game, pnum), dcard['city1'], dcard['city2'])
    #         for p in paths_generator:
    #             print(p)
        
    #     return min_poss_paths + 1 #to avoid zeroing out in vector multiplication
    
    def feature_dcard_points_remaining(self, game, pnum):
        res = 0
        for dcard in self.remaining_dest:
            res += dcard['points']
        return res
    
    def feature_num_traincar_cards_for_dcards_remaining(self, game, pnum):
        return self.num_cards_needed

    def feature_faceup_cards_for_dcards_remaining(self, game, pnum):
        count = 0
        cards_face_up = {k: v for k, v in game.train_cards_face_up.items() if v != 0}
        
        for c in cards_face_up:
            if self.cards_needed[c] > 0:
                count += min(self.cards_needed[c], cards_face_up[c])
        
        return count
    
    def feature_current_points(self, game, pnum):
        return game.players[pnum].points

    def feature_wild_cards_available(self, game, pnum):
        return game.train_cards_face_up["wild"]
    
    def update(self, pnum, game_after_action, game_before_next_turn, reward):
        #ISSUE: others will play between now and then!
        #SOLUTION: evaluate right before action based on previous action
        try:
            assert len(game_before_next_turn.get_possible_moves(pnum)) > 0
        except AssertionError as e:
            print(f"face up {game_before_next_turn.train_cards_face_up}")
            print(f"train deck {sum(game_before_next_turn.train_deck.deck.values())}")
            print(f"destination deck {sum(game_before_next_turn.destination_deck.deck.values())}")
            exit(1)

        possible_nextgame_actions = [a.decide(game_before_next_turn.copy(), pnum) for a in self.agents]
        best_future_q = 0
        for (i, act) in enumerate(possible_nextgame_actions):
            assert act is not None, f"{self.agents[i].__class__.__name__} returned a None action"
            game_after_nextgame_a = game_before_next_turn.copy()
            game_after_nextgame_a.make_move(act.function, act.args)
            future_q = sum([f * w for f, w in zip(self.get_features(game_after_nextgame_a, pnum), self.weights)])
            if best_future_q < future_q:
                best_future_q = future_q
        
        cur_features = self.get_features(game_after_action, pnum)
        difference = reward + self.discount * best_future_q - sum([f * w for f, w in zip(cur_features, self.weights)])
        for i in range(len(self.weights)):
            self.weights[i] += self.alpha * difference * cur_features[i]

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
            return self.agents[rand_agent_idx].decide(game.copy(), pnum)
        else:
            return self.decide(game, pnum) #will copy game
    
    def train_choose_destination_cards(self, moves, game, pnum, num_keep):
        #with probability epsilon, randomly decide between the agents
        #with probabiliity 1-epsilon, just take the real action
        if random.random() < self.epsilon:
            rand_agent_idx = random.randint(0, len(self.agents) - 1)
            return self.agents[rand_agent_idx].choose_destination_cards(moves.copy(), game.copy(), pnum, num_keep)
        else:
            return self.choose_destination_cards(moves, game, pnum, num_keep) #will copy moves/game