import sys
sys.path.insert(0, 'scripts/')

from loadDestinationDeck import *
from loadMap import *
from ttrengine import *
from pathAgent import *
from hungryAgent import *
from oneStepThinkerAgent import *
from longRouteJunkieAgent import *
from QLearningAgent import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import ast

board = Board(loadgraphfromfile("gameContent/usa.txt"))
dest_deck_dict = destinationdeckdict(dest_list=loaddestinationdeckfromfile("gameContent/usa_destinations.txt"), board="usa")

prefix_map_len_3 = {"HLO": [HungryAgent(), LongRouteJunkieAgent(), OneStepThinkerAgent()],
                    "HLP": [HungryAgent(), LongRouteJunkieAgent(), PathAgent()],
                    "HPO": [HungryAgent(), PathAgent(), OneStepThinkerAgent()],
                    "PLO": [PathAgent(), LongRouteJunkieAgent(), OneStepThinkerAgent()]}

prefix_map_len_2 = {"HL": [HungryAgent(), LongRouteJunkieAgent()],
                    "HO": [HungryAgent(), OneStepThinkerAgent()],
                    "HP": [HungryAgent(), PathAgent()],
                    "LO": [LongRouteJunkieAgent(), OneStepThinkerAgent()],
                    "LP": [LongRouteJunkieAgent(), PathAgent()],
                    "OP": [OneStepThinkerAgent(), PathAgent()]}

prefix_map = {"LO": [LongRouteJunkieAgent(), OneStepThinkerAgent()]}#prefix_map_len_2
for prefix in prefix_map:
    #Create agent
    agent = QLearningAgent(prefix_map[prefix])

    #From score_log.txt
    with open(prefix + "_q_values_updated.txt", 'r') as f:
        qvals = f.read()
    agent.qvalues = ast.literal_eval(qvals)

    #Summary statistic reporting
    #Can compute wins from sum of all 1 values in places list 
    ag_dict = {"scores": [], "places": [], "winner list": {}, "winner diff": [], "total dest cards completed": 0, "winner total dest cards completed": []}
    ag_map = {0: "QLearningAgent", 1: "HungryAgent", 2: "OneStepThinkerAgent", 3: "LongRouteJunkieAgent"}
    #test it out!
    num_games = 100

    game_no = 0
    while game_no < num_games:
        player_list = [Player(hand=emptyCardDict(), number_of_trains=45, points=0) for i in range(0,4)]
        game_object = Game(board=board.copy(), point_table=point_table(), destination_deck=dest_deck_dict.copy(), train_deck=make_train_deck(number_of_color_cards=12, number_of_wildcards=14), players=player_list, current_player=0, variants=[3, 2, 3, 1, True, False, False, False, False, False, 4, 5, 2, 3, 2, 10, 15, 2, False])
        gh = GameHandler(game=game_object, agents=[agent, HungryAgent(), OneStepThinkerAgent(), LongRouteJunkieAgent()], filename="test")
        gh.train = False
        gh.ql_indices = set([0])
        gh.play(runnum=game_no, save=False)
        if gh.run_failure:
            print(f"Failure detected, redoing run {game_no}")
            continue
        print(f"--------------------DONE WITH GAME {game_no}--------------------dcards this run {game_object.getNumCompletedDCards(0)}")

        ag_dict["scores"].append(player_list[0].points)
        ag_dict["places"].append(game_object.get_place(0))
        winners = game_object.winner()
        one_winner = winners[0]
        for w in winners:
            w_name = ag_map[w]
            if w_name in ag_dict["winner list"]:
                ag_dict["winner list"][w_name] += 1 / len(winners)
            else:
                ag_dict["winner list"][w_name] = 1 / len(winners)

        ag_dict["winner diff"].append(player_list[one_winner].points - player_list[0].points)
        ag_dict["total dest cards completed"] += game_object.getNumCompletedDCards(0)
        ag_dict["winner total dest cards completed"].append(game_object.getNumCompletedDCards(one_winner))

        for ag2 in gh.agents_reporting:
            if ag2 in ag_dict:
                ag_dict[ag2] += gh.agents_reporting[ag2]
            else:
                ag_dict[ag2] = gh.agents_reporting[ag2]

        game_no += 1
    print("--------------------DONE WITH SET OF GAMES--------------------")

    with open('score_log.txt', 'a') as f:
        original_stdout = sys.stdout
        try:
            sys.stdout = f
            print("Agent results for agents", prefix)
            print(ag_dict)
        finally:
            sys.stdout = original_stdout
        