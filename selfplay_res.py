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

board = Board(loadgraphfromfile("gameContent/usa.txt"))
dest_deck_dict = destinationdeckdict(dest_list=loaddestinationdeckfromfile("gameContent/usa_destinations.txt"), board="usa")
agent_lst = [ApproximateQLearningAgent(), ApproximateQLearningAgent(), ApproximateQLearningAgent(), ApproximateQLearningAgent()]

agent_lst[0].weights = [0.6868523713141634, 4.712487177776475, 1.2127375052906157, 4.936765044314656, 1.4308975484045723, 1.1404473530730612, -1.4134757585127393, -0.20884082768968587]
agent_lst[1].weights = [432.36893958958916, 5015.708485000397, 279.7277611081693, 1345.151267239236, -1092.167843686182, 367.25963452082203, -554.9345167749406, 38.032121724373766]
agent_lst[2].weights = [94.8100028145641, 1742.0800168370417, 28.302515492887345, 851.6913425634069, -1344.7193923079008, 512.9818240118922, -189.9559202053302, -30.87597776296886]
agent_lst[3].weights = [0.21103135028562742, 0.5410644853046008, 0.528287609990957, 0.5766041615417561, 0.024052242855947356, 0.4406086835588685, 0.06708354310520917, -0.13408971385987764]

#test it out!
test_score_record = [[], [], [], []]
test_winner_diff_record = [[], [], [], []]
for iter in range(50):
    for i in range(len(agent_lst)):
        player_list = [Player(hand=emptyCardDict(), number_of_trains=45, points=0) for i in range(0,4)]
        game_object = Game(board=board.copy(), point_table=point_table(), destination_deck=dest_deck_dict.copy(), train_deck=make_train_deck(number_of_color_cards=12, number_of_wildcards=14), players=player_list, current_player=0, variants=[3, 2, 3, 1, True, False, False, False, False, False, 4, 5, 2, 3, 2, 10, 15, 2, False])
        gh = GameHandler(game=game_object, agents=[agent_lst[i], HungryAgent(), OneStepThinkerAgent(), LongRouteJunkieAgent()], filename="test")
        gh.train = False
        gh.play(runnum=i, save=False)

        #print results
        #print(f"Scoring Breakdown for agent {i} (player 0)")
        #gh.game.print_scoresheet()
        winner = game_object.winner()[0]
        test_score_record[i].append(player_list[0].points)
        test_winner_diff_record[i].append(player_list[winner].points - player_list[0].points)

for i in range(len(test_score_record)):
    print(f"Agent {i} scored {np.mean(np.arr(test_score_record[i]))} points on average, losing to the winner by {np.mean(np.arr(test_winner_diff_record[i]))} points on average")