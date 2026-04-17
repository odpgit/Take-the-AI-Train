import sys
sys.path.insert(0, 'scripts/')

from loadDestinationDeck import *
from loadMap import *
from ttrengine import *
from pathAgent import *
from hungryAgent import *
from oneStepThinkerAgent import *
from longRouteJunkieAgent import *
from approximateQLearningAgent import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import ast

board = Board(loadgraphfromfile("gameContent/usa.txt"))
dest_deck_dict = destinationdeckdict(dest_list=loaddestinationdeckfromfile("gameContent/usa_destinations.txt"), board="usa")

#Create agents to compare
sp_agent = ApproximateQLearningAgent()
no_sp_agent = ApproximateQLearningAgent()
agent_lst = [no_sp_agent] #[sp_agent, no_sp_agent]

#name them for ease of reference
sp_agent.name = "Selfplay agent"
no_sp_agent.name = "Non-selfplay agent"

#From score_log.txt
with open("q_values.txt", 'r') as f:
    qvals = f.read()
no_sp_agent.qvalues = ast.literal_eval(qvals)

#Summary statistic reporting
#Can compute wins from sum of all 1 values in places list 
orig_dict = {"scores": [], "places": [], "winner diff": [], "total dest cards completed": 0, "winner total dest cards completed": []}
agent_dicts = {sp_agent.name: copy.deepcopy(orig_dict), no_sp_agent.name: copy.deepcopy(orig_dict)}

#test it out!
num_games = 100
for (i, ag) in enumerate(agent_lst):
    cur_dict = agent_dicts[ag.name]
    game_no = 0
    while game_no < num_games:
          player_list = [Player(hand=emptyCardDict(), number_of_trains=45, points=0) for i in range(0,4)]
          game_object = Game(board=board.copy(), point_table=point_table(), destination_deck=dest_deck_dict.copy(), train_deck=make_train_deck(number_of_color_cards=12, number_of_wildcards=14), players=player_list, current_player=0, variants=[3, 2, 3, 1, True, False, False, False, False, False, 4, 5, 2, 3, 2, 10, 15, 2, False])
          gh = GameHandler(game=game_object, agents=[ag, HungryAgent(), OneStepThinkerAgent(), LongRouteJunkieAgent()], filename="test")
          gh.train = False
          gh.aql_indices = set([0])
          gh.play(runnum=game_no * (i+1), save=False)
          if gh.run_failure:
               print(f"Failure detected, redoing run {game_no}")
               continue
          print(f"--------------------DONE WITH GAME {game_no}--------------------dcards this run {game_object.getNumCompletedDCards(0)}")

          cur_dict["scores"].append(player_list[0].points)
          cur_dict["places"].append(game_object.get_place(0))
          winner = game_object.winner()[0]
          cur_dict["winner diff"].append(player_list[winner].points - player_list[0].points)
          cur_dict["total dest cards completed"] += game_object.getNumCompletedDCards(0)
          cur_dict["winner total dest cards completed"].append(game_object.getNumCompletedDCards(winner))

          for ag2 in gh.agents_reporting:
               if ag2 in cur_dict:
                    cur_dict[ag2] += gh.agents_reporting[ag2]
               else:
                    cur_dict[ag2] = gh.agents_reporting[ag2]

          game_no += 1
    print("--------------------DONE WITH SET OF GAMES--------------------")

for ag in agent_lst:
    with open('score_log.txt', 'a') as f:
        original_stdout = sys.stdout
        try:
            sys.stdout = f
            print("Agent results for agent", ag.name)
            print(agent_dicts[ag.name])
        finally:
            sys.stdout = original_stdout
    