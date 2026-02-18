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

board = Board(loadgraphfromfile("gameContent/usa.txt"))
dest_deck_dict = destinationdeckdict(dest_list=loaddestinationdeckfromfile("gameContent/usa_destinations.txt"), board="usa")
agent_lst = [ApproximateQLearningAgent(), ApproximateQLearningAgent(), ApproximateQLearningAgent(), ApproximateQLearningAgent()]
epsilon_start = 1
for a in agent_lst:
    a.epsilon = epsilon_start

num_training_sessions = 10000
epsilon_target = 0.05
when_reach_target = 0.65 * num_training_sessions

for game_no in range(num_training_sessions):
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

    gh.play(runnum=game_no, save=False)
    #record points somehow

#record weights somehow