from agent import Agent
import networkx as nx
from ttrengine import emptyCardDict

#Question: how to handle storing feature weights?
#   Options: include training function that must be run every time before actually running agent for gameplay
#   DO THIS  include training function that only needs to be run once (and stores results with pickle to load in at __init__)
#Question: which features to weigh?
#             Which features are avilable in game class?
#               - Board (graph)
#               - Point Table
#               - Destination Deck
#               - Train Deck
#               - Train Cards Face Up
#               - Number of Players: self.players[self.current_player].number_of_trains
#               - Current Player
#               - Players Choosing Destination Cards: set to True in init
#               - Who Went First: random. Not useful.
#               - Last Turn Player
#               - Moves Reference: just a lookup for specific move functions. Not useful.
#               - Rules Variants
#               - Number of Current Draws
# returnCurrentPoints function?
# findMaxWeightSumForNode function - pass in a player and start node, can calculate longest route
#           Expensive - need to call for each start node. Store longest route for each player locally during game, update with moves?

#Glenn: features involving the action - know what action each of the four agents would pick
#       "this action scores points for me, completes card, claims route, picks useful traincard etc."

#   TENTATIVE LIST
#   (0) Number of players in game (if not fixed)
#   (1) Minimum number of trains left among all players
#   (2) Number of trains behind player with longest route
#   (3) Minimum number of possible paths to complete a destination card
#   (4) Sum of points remaining to complete all destination cards in hand
#   (5) TRACTABLE? Number of train car cards needed to complete any of the destination cards in hand
#   (6) TRACTABLE? Number of face-up train car cards that let you complete a route needed for a destination card 
#   (TODO): Current points compared to others
#   (TODO): Wild cards available
#   (7) Turn number

#   General strategies of each agent
#   (Hungry Agent)      Accumulates destination cards until a threshold is reached. Keeps destination cards that maximize points scored/train tokens needed
#                       Generates (and recalculates) list of routes needed to connect any city in any card to any other city in any card
#                       Evaluates which train cards are needed, builds priority list based on how many of each are required
#                       For each turn: prioritizes drawing destination cards, then claiming routes, then drawing cards
#   (Route/Path Agent)  Only uses destination cards from the start of the game
#                       Priority queue of routes needed to complete destination cards.
#                           Score is sum of point value and 2x point value (b/c potential penalty) of the associated destination card
#                       For each turn: prioritizes claiming any of the top routes in the priority queue, then draw train car cards of highest priority route
#   (One Step Agent)    Complete one destination, one route at a time
#                       Selects destination cards like Hungry, but redraws when all complete
#                       Prioritize cards not completed worth most points. Among those, prioritizes least expensive (# cards needed to claim it and # cards agent holding) routes
#                       Tries to claim route with highest priority, if not then draw cards of that route's color.
#                           Else, draws destination cards if >= 5 trains left. Else, claims largest route possible with # trains left
#   (Long Route Agent)  Selects destination cards like One Step/Hungry, no redraw.
#                       Among all routes needed for destination cards, prioritizes longer routes, gray routes, routes that require fewest additional draws
#                           If destination cards complete, selects among all unclaimed routes of size 3 or more
# Make sure to: train against rotating set of agents
#           2-player game against each alone equal # of times? 4-player game against 2 of 3 rotating?        
#         
class ApproximateQLearningAgent(Agent):
    def __init__(self, numPlayers):
        self.longest_route_by_player = [0] * numPlayers
        self.jgraph = None
        self.remaining_dest = []
        self.paths_planned = {}
        self.cards_needed = emptyCardDict()
        self.num_cards_needed = 0
        self.turn_number = 0

    def weight_function(zero_edges):
        def fn(u, v, d):
            if (u, v) in zero_edges or (v, u) in zero_edges:
                return 0
            return d.get("weight", 1)
        return fn
    
    def decide(self, game, pnum):
        #update turn number
        self.turn_number += 1

        #update joint graph/remaining destinations storage
        self.jgraph = self.joint_graph(game, pnum)
        self.remaining_dest = self.destinations_not_completed(game, pnum)

        #update paths planned with shortest available path to each remaining dcard
        claimed_edges = set(game.player_graph(pnum).edges())

        #figure out what color cards are needed to complete each of these paths
        #note: the two below variables are different since routes with multiple colors to claim show up 2x in self.cards_needed
        self.cards_needed = emptyCardDict()
        self.num_cards_needed = 0
        for d in self.remaining_dest:
            #this shortest path will include this player's already-claimed routes!
            self.paths_planned[d] = nx.shortest_path(self.jgraph, d['city1'], d['city2'], weight=self.weight_function(claimed_edges))
            for i in range(0, len(self.paths_planned[d]) - 1):
                node1 = self.paths_planned[d][i]
                node2 = self.paths_planned[d][i+1]
                #first, check if route already claimed by this player
                if node1 in game.player_graph(pnum)[node2] or node2 in game.player_graph(pnum)[node1]:
                    #already claimed - don't need any more cards
                    continue

                #otherwise, find pairing on game board
                edgelist = game.board.get_free_connection(node1, node2)

                self.num_cards_needed += edgelist[0]['weight']

                #decision: if route has multiple ways to claim it, add to both
                for e in edgelist:
                    #note: graph has colors uppercase but cards_needed has colors lowercase
                    #note: if e['color'] is gray, will just add gray to self_cards_needed. That's OK.
                    self.cards_needed[e['color'].lower()] += e['weight']

        #update longest route by player
        for pnum in range(len(self.longest_route_by_player)):
            player_graph = game.player_graph(pnum)
            longest_by_node = [self.findMaxWeightSumForNode(player_graph, v, []) for v in player_graph.nodes()]
            self.longest_route_by_player[pnum] = max(longest_by_node) if len(longest_by_node) > 0 else 0

        #consult features to make decision (AFTER above updates)

    def feature_num_players(self, game, pnum):
        return game.number_of_players()
    
    def feature_minimum_trains_left(game, pnum):
        min_trains = float('inf')
        for p in game.players:
            if p.number_of_trains < min_trains:
                min_trains = p.number_of_trains
        return min_trains
    
    def feature_longest_route(self, game, pnum):
        own_longest_route = self.longest_route_by_player[pnum]
        overall_longest_route = max(self.longest_route_by_player)
        return overall_longest_route - own_longest_route + 1 #to avoid zeroing out in vector multiplication

    def feature_minimum_possible_dcard_paths(self, game, pnum):
        if len(self.remaining_dest) == 0:
            return 0 #zero out in vector multiplication
        
        min_poss_paths = float('inf')
        for dcard in self.remaining_dest:
            #get possible paths to dcard
            paths_generator = nx.all_simple_paths(self.joint_graph, dcard.city1, dcard.city2)
            min_poss_paths = min(min_poss_paths, len(paths_generator))
        
        return min_poss_paths + 1 #to avoid zeroing out in vector multiplication
    
    def feature_dcard_points_remaining(self, game, pnum):
        res = 0
        for dcard in self.remaining_dest:
            res += dcard.points
        return res
    
    def feature_num_traincar_cards_for_dcards_remaining(self, game, pnum):
        return self.num_cards_needed

    def feature_faceup_cards_for_dcards_remaining(self, game, pnum):
        count = 0
        cards_face_up = {k: v for k, v in game.train_cards_face_up().items() if v != 0}
        
        for c in cards_face_up:
            if self.cards_needed[c] > 0:
                count += min(self.cards_needed[c], cards_face_up[c])
        
        return count

    def feature_turn_number(self, game, pnum):
        return self.turn_number
    
    def feature_current_points(self, game, pnum):
        return game.players[pnum].points

    def feature_wild_cards_available(self, game, pnum):
        return game.train_cards_face_up["wild"]