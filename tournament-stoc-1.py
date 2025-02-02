"""
    run tournament with all rule-based agents and stochastic agents.
    stoc1 agents play C or D with certain probabilities.
"""
import itertools
from datetime import datetime

import pandas as pd

from tournament.agents.agents import AGENTS
from tournament.agents.probabilistic.random_stoc1 import STOC_AGENTS, STOC_HYPERPARAMS
from tournament.tournament import RoundRobinTournament


def main():
    tournament = RoundRobinTournament(AGENTS + STOC_AGENTS)
    scores, times = tournament.play(
        continuation_probability=0.99654, repetitions=50, jobs=12
    )

    data = [
        {
            "id": agent.__name__,
            **STOC_HYPERPARAMS[agent.__name__],
            "tn_score": sum(scores[agent]) / len(scores[agent]),
            "tn_time": sum(times[agent]) / len(times[agent]),
        }
        if agent.__name__ in STOC_HYPERPARAMS
        else {
            "id": agent.__name__,
            "tn_score": sum(scores[agent]) / len(scores[agent]),
            "tn_time": sum(times[agent]) / len(times[agent]),
        }
        for agent in scores
    ]

    df = pd.DataFrame(data)
    df.to_csv(
        f"results/stoc-1-tournament-{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv"
    )


if __name__ == "__main__":
    main()
