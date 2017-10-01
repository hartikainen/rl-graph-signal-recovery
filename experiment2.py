from envs import GraphSamplingEnv
from agents import BaseAgent

def train_test_agent():
  M = 10
  env = GraphSamplingEnv(max_samples=M)

  num_train_graphs = 10

  agent = BaseAgent(env=env)
  agent.learn(num_train_graphs)
  agent.test()

if __name__ == "__main__":
  train_test_agent()
