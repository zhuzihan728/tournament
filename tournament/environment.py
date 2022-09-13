from datetime import datetime
from typing import List, Type

from tournament.action import Action
from tournament.agent import Agent, TrainableAgent
from tournament.match import Match


class Environment:
    """
        An environment setup for training a learning agent.
    """
    
    
    def __init__(self, silent: bool = False) -> None:
        """

        Args:
            silent (bool, optional): to show logs or not
        """
        self.silent = silent 
        self.counts = {Action.COOPERATE: 0, Action.DEFECT: 0}      # stores the total numbers of C and D during the training
        self.epoch_count = {Action.COOPERATE: 0, Action.DEFECT: 0} # stores the total numbers of C and D for every epoch
        self.epoch_counts = []                                     # stores epoch_count for all epochs
        self.normalised_epoch_counts = []                          # stores normalized_epoch_counts {C%, D%} for all epochs
        self.rewards = []                                          # stores reward every turn 
        self.metric_history = []                                   # stores the metrics for all batches

    def _play_training_match(
        self,
        trainee: TrainableAgent,
        opponent: Agent,
        continuation_probability: float,
        limit: int,
        noise: float,
    ):
        """
        
        One match with a rule-based agent in the environment.

        Args:
            trainee (TrainableAgent): the learning agent
            opponent (Agent): the rule-based opponent
            
            Args for match(trainee, opponent):
            continuation_probability (float): the prob to continue playing match
            limit (int): the max turns in match
            noise (float): the prob to mutate one's action.
        """
        trainee.notify_prematch()
        for moves, scores, rewards in Match(trainee, opponent).play_moves(
            continuation_probability=continuation_probability,
            limit=limit,
            noise=noise,
        ):
            self.counts[moves[0]] += 1
            self.epoch_count[moves[0]] += 1
            self.rewards.append(rewards[0])
            trainee.update(moves, scores, rewards)

        trainee.notify_postmatch()
        self.metric_history.append(trainee.metric)

    def _play_epoch(
        self,
        trainee: TrainableAgent,
        continuation_probability: float = 1,
        limit: int = 10000,
        noise: float = 0,
        repetitions: int = 1,
    ):
        """
        an epoch for training, 
        to be overridden

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def train(
        self,
        trainee: TrainableAgent,
        continuation_probability: float = 1,
        limit: int = 100000,
        noise: float = 0,
        repetitions: int = 1,
        epochs: int = 1,
    ) -> None:
        """
        The whole training process.
        Playing a number of epochs.
        One epoch is to match with every other rule-based agents for a number of repetitions.

        Args:
            trainee (TrainableAgent): the learning agent
                Args for match(trainee, opponent):
                    continuation_probability (float): the prob to continue playing match
                    limit (int): the max turns in match
                    noise (float): the prob to mutate one's action.
                repetitions: the number of matches against one agent
                
            epochs (int, optional): number of epochs. Defaults to 1.
        """
        trainee.setup()

        if not self.silent:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Commencement of training.")

        for i in range(epochs):
            self.epoch_count = {Action.COOPERATE: 0, Action.DEFECT: 0} # initialize action counter for this epoch

            self._play_epoch(
                trainee, continuation_probability, limit, noise, repetitions # play one epoch
            )

            s = sum(self.epoch_count.values())  # calculate the number of actions in total of this epoch (this is needed as the continuation_probability is not 1)
            self.epoch_counts.append(self.epoch_count)   # add the count {C:_, D:_} for this epoch to self.epoch_counts
            
            # calculate the portions of C and D {C:_, D:_} from this epoch, and add it to normalised_epoch_counts
            self.normalised_epoch_counts.append(    
                {a: self.epoch_count[a] / s for a in self.epoch_count}  
            )

            if not self.silent:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Completed epoch {i + 1}: {trainee.metric}"
                )

        trainee.teardown()
