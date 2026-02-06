from agent import Agent
import networkx as nx

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
        for d in self.remaining_dest:
            self.paths_planned[d] = nx.shortest_path(self.jgraph, d['city1'], d['city2'], weight=self.weight_function(claimed_edges))

        #update longest route by player somehow

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
        pass 

    def feature_faceup_cards_for_dcards_remaining(self, game, pnum):
        #game.train_cards_face_up
        count = 0
        for path in self.paths_planned:
            pass

    def feature_turn_number(self, game, pnum):
        return self.turn_number
    
    def feature_current_points(self, game, pnum):
        return game.players[pnum].points

    def feature_wild_cards_available(self, game, pnum):
        return game.train_cards_face_up["wild"]