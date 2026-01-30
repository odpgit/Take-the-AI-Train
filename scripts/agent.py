from abc import ABC, abstractmethod
import networkx as nx

class Agent(ABC):
	@abstractmethod
	def decide(self, game, pnum):
		"Must be implemented by subclass"
		pass
	
	def free_routes_graph(self, pnum, graph, number_of_players, min_weight_edge=0):
		G = nx.MultiGraph()
		visited_nodes = []
		
		for node1 in graph:
			for node2 in graph[node1]:
				if node2 not in visited_nodes:
					locked = False
					for edge in graph[node1][node2]:
						if number_of_players < 4:  #################### CHECK THIS FOR SWITZERLAND!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
							if graph[node1][node2][edge]['owner'] != -1:
								locked = True
						else:
							#if you already own one side of a double route, can't claim other
							if graph[node1][node2][edge]['owner'] == pnum:
								locked = True

					if not locked:
						for edge in graph[node1][node2]:
							if graph[node1][node2][edge]['owner'] == -1 and graph[node1][node2][edge]['weight'] >= min_weight_edge:
								G.add_edge(node1, node2, weight=graph[node1][node2][edge]['weight'], color=graph[node1][node2][edge]['color'], underground=graph[node1][node2][edge]['underground'], ferries=graph[node1][node2][edge]['ferries'], owner=-1)

			visited_nodes.append(node1)
		
		return G
	
	def joint_graph(self, game, pnum, min_weight_edge=0):
		free_connections_graph = self.free_routes_graph(pnum, game.board.graph, game.number_of_players, min_weight_edge)
		player_edges = game.player_graph(pnum).edges()
		
		joint_graph = free_connections_graph
		for edge in player_edges:
			joint_graph.add_edge(edge[0], edge[1], weight=0, color='none', owner=pnum, underground=False, ferries=0)

		return joint_graph