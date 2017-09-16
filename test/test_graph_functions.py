from envs import GraphSampling2
from utils import load_graph
from algorithms.recovery import sparse_label_propagation
from algorithms.sampling import RandomWalkSampling
from graph_functions import normalized_mean_squared_error, total_variance

graph = load_graph("./data/graphs/out2.json")
random_walker = RandomWalkSampling("./data/graphs/out2.json")
sampling_result = random_walker.run()
sampling_set = sampling_result['sampling_set']

recovered_graph = sparse_label_propagation(graph, list(sampling_set))
nmse = normalized_mean_squared_error(graph, recovered_graph)
tv = total_variance(graph, recovered_graph)
