# Reinforcement Learning for Graph Signal Recovery
This project implements a reinforcement learning approach for graph sampling in
graph signal recovery problems. Reinforcement learning policy is trained to
walk a graph and sample nodes. Sparse label propagation is applied to the
sampled nodes to recover the graph signal. Tools for the training and
evaluation of rl algorithms for the sampling problem are provided.

## Graph Sampling RL Environment
Graph sampling is formulated as a reinforcement learning problem by defining an
agent that is capable of moving along the graph edges and choosing nodes to
sample, guided by observations and rewards from the environment. The actions of
the agent are moving along the graph edges and sampling nodes. The reward is
defined in terms of the error of the recovered graph signal and the
observations summarize local characteristics of the graph surrounding the
agent. The environment implements the [OpenAI
Gym](https://github.com/openai/gym) environment API, which makes it easy to use
various high performance algorithms with it. The graph sampling environment is
defined in `envs/graph_sampling_env.py`.

The current implementation of the sampling environment is only an initial draft
of the ideas and thus, although it's fully functional, it's not likely to work
for training efficient sampling agents.

### The Graph
At the start of each training run, the environment generates a random
assortative planted partition model. The graph generator is implemented in
`generate_appm.py`. The idea is to train the agent on random graphs forcing it
to learn to generalize across different instances of the graphs.

### Actions
The agent picks actions to move along the edges of the graph and sample
nodes. The high performing reinforcement learning models are limited to fixed
size inputs and outputs, but APPM and other types of graphs may have arbitrary
numbers of nodes and edges.  The degree of the nodes may vary as well. In
order to fix the size of the the action space, we reduce the movement actions
into binary choices following ideas presented in Section 5.2 of [Information
Theory of Decisions and
Actions](https://link.springer.com/chapter/10.1007/978-1-4419-1452-1_19).
Specifically, the environment keeps track of the node the agent is located in
and the currently selected edge. The agent always faces a binary choice of
either moving along the selected edge or advancing to the next edge.

In addition to the two movement actions, the agent can choose to sample the
current node. After sampling the node, the node is added to the sampling set,
reward is computed and the agent is transported to a new random node in the
graph. If the sampling budget is reached, the training episode ends and no
further rewards are available to the agent.

### Observations
The observations summarize characteristics of the graph. Currently, the
observations consist of the local clustering coefficients and degrees of the
nodes close to the agent. To keep the observation size fixed, the statistics of
the neighboring nodes are represented through min, max and mean. In addition,
the clustering coefficient and the degree of the current node and the next node
are presented as is.

### Reward
Each time a node is sampled, the agent receives a reward. After sampling a new
node, the signal recovery algorithm is run. The error of the recovered signal
is used to compute the reward.

### Notes
The current implementation of the environment is unfinished and unlikely to
yield strong sampling performance.

* Randomizing the agent location after each sample is obtained seems like it's
  beating the purpose of the learning algorithm to some extent. If we are able
  to provide the agent with a more global view to the graph, this limitation
  can hopefully be lifted.
* The observations are subject to change in the future iterations of the
  environment.
* The reward computation is work in progress and will be rethought and
  reimplemented soon.
* There are multiple ways to split the actions down into a fixed set of
  actions. The other ways are currently low priority on the list of things to
  experiment with.

## Running RL in the Graph Sampling Environment
Reinforcement learning algorithms implemented in the [OpenAI
Baselines](https://github.com/openai/baselines/) can be used directly with the
graph sampling enviroment. In `experiment2.py` code is provided for running the
Deep Q-Learning agent in the Graph Sampling Environment. Experimental results
will be made available following progress with the environment.

## Random Walk Sampling
Random walk sampling as presented in [Random Walk Sampling for Big Data over
Networks](https://arxiv.org/abs/1704.04799v1) is implemented in
`algorithms/sampling/random_walk_sampling.py`. The results in the paper for
signal recovery on generated appms with random walk sampling in can be
reproduced by running `experiment1.py` like follows:

```
python ./experiment1.py --step=graph_generate --cluster_sizes 10 20 30 40 --p=0.3 --q=0.04 --num_graphs=10000 --results_base=./data/experiment1/graphs

python ./experiment1.py --step=sampling --Ms 10 --Ls 20 40 80 160 320

python ./experiment1.py --step=recovery
```

The following results were obtained:

Sampling Budget | Length of Walk | Mean NMSE   | Std NMSE
----------------|----------------|-------------|-----------
10              | 20             | 0.315172    | 0.24432588
10              | 40             | 0.314349    | 0.24225918
10              | 80             | 0.31593065  | 0.24468789
10              | 160            | 0.31569943  | 0.24466505
10              | 320            | 0.31578363  | 0.24517507
