import random
from datetime import datetime
from typing import List, Type

from tournament.agent import Agent, TrainableAgent
from tournament.environment import Environment


class MultipleRuleBasedAgentEnvironment(Environment):
    """
        The training environment with multiple rule-based agents.
    Args:
        Environment (_type_): implement the Environment class.
    """
    def __init__(self, agents: List[Type[Agent]], silent: bool = False) -> None:
        
        super().__init__(silent)

        self.agents = agents

    def _play_epoch(
        self,
        trainee: TrainableAgent,
        continuation_probability: float = 1,
        limit: int = 10000,
        noise: float = 0,
        repetitions: int = 1,
    ):
        """
        one epoch for training a learning based agent.
        one epoch is matching with every rule-based opponents for a number of repetitions

        Args:
            trainee (TrainableAgent): the agent to train
            continuation_probability (float, optional): the probability of continue to play in a match. Defaults to 1.
            limit (int, optional): number of turns of a match. Defaults to 10000.
            noise (float, optional): the probability an agent flip the action. Defaults to 0.
            repetitions (int, optional): how many matches to play againest one agent. Defaults to 1.
        """
        
        # randomly pick the opponent until all are seen.
        for opponent in random.sample(self.agents, len(self.agents)):
            # print(
            #     f"[{datetime.now().strftime('%H:%M:%S')}] Training against {opponent.__name__}"
            # )
            # the learner plays one full game with a rule-based agent.
            for _ in range(repetitions):
                self._play_training_match(
                    trainee, opponent(), continuation_probability, limit, noise
                )
