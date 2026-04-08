import sys
sys.path.insert(0, 'scripts/')
import io

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
import time

board = Board(loadgraphfromfile("gameContent/usa.txt"))
dest_deck_dict = destinationdeckdict(dest_list=loaddestinationdeckfromfile("gameContent/usa_destinations.txt"), board="usa")
agent_lst = [ApproximateQLearningAgent(), ApproximateQLearningAgent(), ApproximateQLearningAgent(), ApproximateQLearningAgent()]

train_score_record = [[], [], [], []]
epsilon_start = 1
for a in agent_lst:
    a.epsilon = epsilon_start

num_training_sessions = 5000
epsilon_target = 0.05
when_reach_target = 0.65 * num_training_sessions

game_no = 0
while game_no < num_training_sessions:
    if game_no >= when_reach_target:
        epsilon_this_iter = epsilon_target
    else:
        epsilon_this_iter = ((epsilon_target - epsilon_start) / when_reach_target) * game_no + epsilon_start
   
    for a in agent_lst:
        a.reinitialize_vars()
        a.epsilon = epsilon_this_iter
    
    player_list = [Player(hand=emptyCardDict(), number_of_trains=45, points=0) for i in range(0,4)]
    game_object = Game(board=board.copy(), point_table=point_table(), destination_deck=dest_deck_dict.copy(), train_deck=make_train_deck(number_of_color_cards=12, number_of_wildcards=14), players=player_list, current_player=0, variants=[3, 2, 3, 1, True, False, False, False, False, False, 4, 5, 2, 3, 2, 10, 15, 2, False])
    gh = GameHandler(game=game_object, agents=agent_lst, filename="test")
    gh.train = True
    gh.aql_indices = set(range(0, 4))

    start = time.time()
    gh.play(runnum=game_no, save=False)
    print (f"Game no. {game_no} took {gh.turn_count} turns ({(time.time() - start):.2f} seconds)")

    #rerun this game number if the run was not successful 
    #record points if the run was successful
    if gh.run_failure:
        print(f"Failure detected, redoing run {game_no}")
    else:
        for i in range(len(player_list)):
            train_score_record[i].append(player_list[i].points)
        game_no += 1

#test it out!
test_score_record = []
for i in range(len(agent_lst)):
    player_list = [Player(hand=emptyCardDict(), number_of_trains=45, points=0) for i in range(0,4)]
    game_object = Game(board=board.copy(), point_table=point_table(), destination_deck=dest_deck_dict.copy(), train_deck=make_train_deck(number_of_color_cards=12, number_of_wildcards=14), players=player_list, current_player=0, variants=[3, 2, 3, 1, True, False, False, False, False, False, 4, 5, 2, 3, 2, 10, 15, 2, False])
    gh = GameHandler(game=game_object, agents=[agent_lst[i], HungryAgent(), OneStepThinkerAgent(), LongRouteJunkieAgent()], filename="test")
    gh.train = False
    gh.aql_indices = set()
    gh.play(runnum=game_no + i + 1, save=False)

    #print results
    print(f"Scoring Breakdown for agent {i} (player 0)")
    gh.game.print_scoresheet()
    #log results
    with open('score_log.txt', 'a') as f:
        original_stdout = sys.stdout
        try:
            sys.stdout = f
            print(f"Scoring Breakdown for agent {i} (player 0)")
            gh.game.print_scoresheet()
        finally:
            sys.stdout = original_stdout

#record weights somehow
with open('score_log.txt', 'a') as f:
    original_stdout = sys.stdout
    try:
        sys.stdout = f
        print("Agent weights")
        for i in range(len(agent_lst)):
            print(i, agent_lst[i].weights)
    finally:
        sys.stdout = original_stdout

#score curve
train_score_record = [np.array(arr) for arr in train_score_record]

train_x = np.arange(num_training_sessions)

for i in range(len(train_score_record)):
    plt.clf()
    plt.plot(train_x, train_score_record[i])
    plt.title(f"Agent {i}'s training score progression")
    plt.savefig(f"Agent {i} training score progression.png")
