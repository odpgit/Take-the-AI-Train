import sys, os
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root_dir, 'scripts'))

from loadDestinationDeck import *
from loadMap import *
from ttrengine import *
from pathAgent import *
from hungryAgent import *
from oneStepThinkerAgent import *
from longRouteJunkieAgent import *
from QLearningAgent import *
import ast

board = Board(loadgraphfromfile("gameContent/usa.txt"))
dest_deck_dict = destinationdeckdict(dest_list=loaddestinationdeckfromfile("gameContent/usa_destinations.txt"), board="usa")

#TODO: can replace with desired heuristic agents
heur_agents = [LongRouteJunkieAgent(), OneStepThinkerAgent()]

#TODO: can replace with filepath to desired Q-values in Python dictionary form
#      See the file from the default filepath for an example
#      Just make sure this matches the above choice of heuristic agents!
qvals = "testing/LO_q_values_updated.txt"

#Create agent
agent = QLearningAgent(heur_agents)

with open(qvals, 'r') as f:
    qvals = f.read()
agent.qvalues = ast.literal_eval(qvals)

#Summary statistic reporting
#Can compute wins from sum of all 1 values in places list 
results_dict = {"scores": [], "places": [], "winner diff": [], "total dest cards completed": 0, "winner total dest cards completed": []}

#test it out!
num_games = 100
game_no = 0
while game_no < num_games:
        player_list = [Player(hand=emptyCardDict(), number_of_trains=45, points=0) for i in range(0,4)]
        game_object = Game(board=board.copy(), point_table=point_table(), destination_deck=dest_deck_dict.copy(), train_deck=make_train_deck(number_of_color_cards=12, number_of_wildcards=14), players=player_list, current_player=0, variants=[3, 2, 3, 1, True, False, False, False, False, False, 4, 5, 2, 3, 2, 10, 15, 2, False])
        gh = GameHandler(game=game_object, agents=[agent, HungryAgent(), OneStepThinkerAgent(), LongRouteJunkieAgent()], filename="ex_test")
        gh.train = False
        gh.ql_indices = set([0])
        gh.play(runnum=game_no, save=False)
        if gh.run_failure:
            print(f"Failure detected, redoing run {game_no}")
            continue
        print(f"--------------------DONE WITH GAME {game_no}--------------------")

        results_dict["scores"].append(player_list[0].points)
        results_dict["places"].append(game_object.get_place(0))
        winner = game_object.winner()[0]
        results_dict["winner diff"].append(player_list[winner].points - player_list[0].points)
        results_dict["total dest cards completed"] += game_object.getNumCompletedDCards(0)
        results_dict["winner total dest cards completed"].append(game_object.getNumCompletedDCards(winner))

        for ag in gh.agents_reporting:
            if ag in results_dict:
                results_dict[ag] += gh.agents_reporting[ag]
            else:
                results_dict[ag] = gh.agents_reporting[ag]

        game_no += 1
print("--------------------DONE WITH SET OF GAMES--------------------")

print("Agent results")
print(results_dict)