from random import random

from tournament.action import Action, flip_action
from tournament.agent import Agent

C, D = Action.COOPERATE, Action.DEFECT

PAYOFF_MATRIX = {
    (C, C): (3, 3),
    (C, D): (0, 5),
    (D, C): (5, 0),
    (D, D): (1, 1),
}


class Match:
    """
        Match is a match of 2 agents to play 2-player IPD
    """
    def __init__(self, agent1: Agent, agent2: Agent) -> None:
        """
        Match two agents

        Args:
            agent1 (Agent): one agent of the match
            agent2 (Agent): one agent of the match
        """
        self.agent1 = agent1
        self.agent2 = agent2

    def _mutate(self, action: Action, noise: float):
        """
        action of an agent, of a probability to flip.

        Args:
            action (Action): the agent's action for this turn.
            noise (float): the probability below which the agent will flip its action.

        Returns:
            Action: the action to take
        """
        if 0 < random() < noise:
            return flip_action(action)

        return action

    def play_moves(self, continuation_probability: float, limit: int, noise: float):
        """
        the two agents play [limit] turns, 
        with a probability [1-continuation_probability] to quit early, 
        and a [noise] to flip their move suggested by their policy

        Args:
            continuation_probability (float): the probability of which the agents will keep playing till limit = the probability the game quits early.
            limit (int): how many turns the pair play
            noise (float): noise for flipping action

        Yields:
            iterator: the moves, current totoal scores, and the payoffs of the two agents every turn
        """
        score1 = 0
        score2 = 0

        history1 = []
        history2 = []

        self.agent1.on_match_start()
        self.agent2.on_match_start()

        i = 0
        while i < limit and (i < 1 or random() < continuation_probability):
            move1 = self._mutate(self.agent1.play_move(history1, history2), noise)
            move2 = self._mutate(self.agent2.play_move(history2, history1), noise)

            increase1, increase2 = PAYOFF_MATRIX[(move1, move2)]
            score1 += increase1
            score2 += increase2

            history1.append(move1)
            history2.append(move2)

            i += 1
            yield (move1, move2), (score1, score2), (increase1, increase2) # use yield to iteratively have results for every turn

        self.agent1.on_match_end()
        self.agent2.on_match_end()

    def play(
        self, continuation_probability: float = 1, limit: int = 10000, noise: float = 0
    ):
        """
            The two agents play a match

        Args:
            continuation_probability (float): the probability of which the agents will keep playing till limit = the probability the game quits early.
            limit (int): how many turns the pair play
            noise (float): noise for flipping action

        Returns:
            scores (int, int): the final scores only
        """
        
        *_, (moves, scores, rewards) = self.play_moves(
            continuation_probability=continuation_probability, limit=limit, noise=noise
        )
        return scores
