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
sp_agent.weights = [np.float64(2.2836743303712512), np.float64(0.3353182479553938), np.float64(0.40885563535285674), np.float64(2.1405829124452604), np.float64(1.8278937641398976), np.float64(1.3922647370098504), np.float64(0.9083352742785792), np.float64(0.13446096456426823), np.float64(13.880437945577768), np.float64(5.269144332074173), np.float64(2.1785564273736617), np.float64(5.9931550688454305), np.float64(6.725395720588794), np.float64(7.063077079619212), np.float64(5.040261490493441), np.float64(0.9943093939344594), np.float64(2.0619411076626806), np.float64(1.2385708267729196), np.float64(0.25613896038308503), np.float64(1.393693217957709), np.float64(1.3876023015927674), np.float64(1.3887267599196074), np.float64(0.5774183435580648), np.float64(0.16201161745980228), np.float64(2.1079029195433137), np.float64(0.9485317909675519), np.float64(0.2941553767119086), np.float64(1.6361029241859333), np.float64(1.4688148404377195), np.float64(1.3697426951581566), np.float64(0.6732449014251688), np.float64(0.14547327957552414)]
no_sp_agent.weights = [107.36585149, -9.21162067,  -77.95171447, -34.36650766,  -24.87353677, 345.08205437, -9.44190556, -277.49580981]

no_sp_agent.weights = np.array(no_sp_agent.weights)

#Summary statistic reporting
#Can compute wins from sum of all 1 values in places list 
orig_dict = {"scores": [], "places": [], "winner diff": [], "total dest cards completed": 0, "winner total dest cards completed": []}
agent_dicts = {sp_agent.name: copy.deepcopy(orig_dict), no_sp_agent.name: copy.deepcopy(orig_dict)}

#test it out!
num_games = 10
for (i, ag) in enumerate(agent_lst):
    cur_dict = agent_dicts[ag.name]
    for game_no in range(num_games):
        player_list = [Player(hand=emptyCardDict(), number_of_trains=45, points=0) for i in range(0,4)]
        game_object = Game(board=board.copy(), point_table=point_table(), destination_deck=dest_deck_dict.copy(), train_deck=make_train_deck(number_of_color_cards=12, number_of_wildcards=14), players=player_list, current_player=0, variants=[3, 2, 3, 1, True, False, False, False, False, False, 4, 5, 2, 3, 2, 10, 15, 2, False])
        gh = GameHandler(game=game_object, agents=[ag, HungryAgent(), OneStepThinkerAgent(), LongRouteJunkieAgent()], filename="test")
        gh.train = False
        gh.aql_indices = set([0])
        gh.play(runnum=game_no * (i+1), save=False)
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
    