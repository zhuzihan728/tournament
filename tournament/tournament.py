from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import chain
from typing import List, Type

from tqdm import tqdm

from tournament.agent import Agent
from tournament.match import Match


def _play_match(a, b, repetitions, continuation_probability, limit, noise):
    """
    two agents play [repetitions] matches.

    Args:
        a (_type_): an agent
        b (_type_): an agent
        repetitions (_type_): how many matches to play
        continuation_probability (_type_): probability to continue play till limit
        limit (_type_): how many turns per match
        noise (_type_): the probability an agent flips actions

    Returns:
        tuple: (type of agent*2, list of scores for every match*2, time it takes)
    """
    scores_a = []
    scores_b = []

    start = datetime.now()
    for _ in range(repetitions):
        score_a, score_b = Match(a, b).play(
            continuation_probability=continuation_probability,
            limit=limit,
            noise=noise,
        )

        scores_a.append(score_a)
        scores_b.append(score_b)
    end = datetime.now()

    return (
        type(a),
        type(b),
        scores_a,
        scores_b,
        (end - start).total_seconds(),
    )


def _play_multiprocessed_match(
    a, b, repetitions, continuation_probability, limit, noise
):
    """
    a copy of _play_match() to use in a thread pool

    """
    return _play_match(a(), b(), repetitions, continuation_probability, limit, noise)


class RoundRobinTournament:
    def __init__(
        self, agents: List[Type[Agent]], instances: List[Agent] = None
    ) -> None:
        """
        Initialize a round robin tournament

        Args:
            agents (List[Type[Agent]]): pre-defined agents, usually rule-based ones
            instances (List[Agent], optional): agent instances, usually trained agents
        """
        self.agents = agents
        self.instances = instances if instances is not None else []
        self.instance_types = [type(x) for x in self.instances]

    def _play_multiprocessed_games(
        self, continuation_probability, limit, noise, repetitions, jobs
    ):
        """
            run the tournament with multiple threads, for pre-defined agents only.

        """
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(
                    _play_multiprocessed_match,
                    a,
                    b,
                    repetitions,
                    continuation_probability,
                    limit,
                    noise,
                )
                for a in self.agents
                for b in self.agents
            ]

            for future in as_completed(futures):
                yield future.result()

    def _play_sequential_games(self, cp, limit, noise, reps):
        """
            run the tournament with single thread, for trained agents against all other agents.
        """
        for a in self.agents:
            for b in self.instances:
                yield _play_match(a(), b, reps, cp, limit, noise)

        for a in self.instances:
            for b in self.agents:
                yield _play_match(a, b(), reps, cp, limit, noise)

        for a in self.instances:
            for b in self.instances:
                yield _play_match(a, b, reps, cp, limit, noise)

    def play(
        self,
        continuation_probability: float = 1,
        limit: int = 10000,
        noise: float = 0,
        repetitions: int = 1,
        jobs: int = 1,
    ):
        """
        play the tournament for all agents.

        Args:
            continuation_probability (float, optional): probability to continue playing. Defaults to 1.
            limit (int, optional): number of turns for each match. Defaults to 10000.
            noise (float, optional): the probability an agent flip action. Defaults to 0.
            repetitions (int, optional): number of matches for every pair. Defaults to 1.
            jobs (int, optional): number of threads. Defaults to 1.

        Returns:
            scores List[List[]]: the scores of the agents against every other agent
            times: time it takes for every pair.
        """
        scores = {agent: [] for agent in self.agents + self.instance_types}
        times = {agent: [] for agent in self.agents + self.instance_types}

        for a, b, scores_a, scores_b, time in tqdm(
            chain(
                self._play_multiprocessed_games(
                    continuation_probability, limit, noise, repetitions, jobs
                ),
                self._play_sequential_games(
                    continuation_probability, limit, noise, repetitions
                ),
            ),
            total=((len(self.agents) + len(self.instances)) ** 2),
            unit="matches",
        ):
            scores[a].extend(scores_a)
            scores[b].extend(scores_b)

            times[a].append(time)
            times[b].append(time)

        return scores, times
