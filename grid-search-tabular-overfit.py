"""
To train against and evaluate with a certain collections of agents.
To see if the agent collection is kind or traitors.

"""

import itertools
from datetime import datetime
from json import dumps

import numpy as np
import pandas as pd

from tournament.agents.axelrod_first import (
    Davis,
    Downing,
    Feld,
    Friedman,
    Graaskamp,
    Grofman,
    Grudger,
    Joss,
    Nydegger,
    Shubik,
    SteinAndRapoport,
    TidemanAndChieruzzi,
    Tullock,
)
from tournament.agents.axelrod_second import (
    Borufsen,
    Champion,
    Leyvraz,
    Black,
    Cave,
    GraaskampAndKatzen,
    Harrington,
    TidemanAndChieruzzi2,
    Weiner,
    White,
    Adams,
)
from tournament.agents.constant import AllC, AllD
from tournament.agents.pavlov import Pavlov
from tournament.agents.q_learning.tabular import TabularQLearner
from tournament.agents.random import RandomAgent
from tournament.agents.tft import (
    TFTT,
    TTFT,
    GenerousTFT,
    GradualTFT,
    OmegaTFT,
    TitForTat,
)
from tournament.gridsearch import train_and_evaluate


class Tabular(TabularQLearner):
    def __init__(
        self,
        lookback,
        discount_rate,
        learning_rate,
        epsilon,
        epsilon_decay,
        decay_limit,
    ) -> None:
        super().__init__()

        self._lookback = lookback
        self._discount_rate = discount_rate
        self._learning_rate = learning_rate
        self.epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._decay_limit = decay_limit


def main(agents):
    print(agents)

    grid = {
        "lookback": [2, 4],
        "epsilon": [0.2],
        "epsilon_decay": [0.0],
        "decay_limit": [0.05],
        "learning_rate": [0.01],
        "discount_rate": [0.95],
    }

    d = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    results = []
    try:
        space = list(itertools.product(*grid.values()))
        size = len(space)
        for i, hyperparameters in enumerate(space):
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H-%M-%S')} | {i + 1}/{size}]",
                *hyperparameters,
                sep="\t",
            )
            result, agent = train_and_evaluate(
                agents,
                Tabular,
                epochs=2500,
                tournament_agents=agents,
                **dict(zip(grid.keys(), hyperparameters)),
            )
            results.append(result)
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H-%M-%S')} | {i + 1}/{size}]",
                f"COOP%={results[-1]['tr_cooperation_percentage']}",
                f"LOSS={results[-1]['tr_final_loss']}",
                f"RANK={results[-1]['tn_rank']}",
                f"SCORE={results[-1]['tn_mean_score']}",
                sep="\t",
            )
            if result["tn_mean_score"] > 750 or result["tn_rank"] > 26:
                np.savez_compressed(
                    f"models/tabular-overfit/{d} - {i} - {result['tn_mean_score']} - {result['tn_rank']}.npz",
                    q_table=agent._q_table,
                )
                with open(
                    f"models/tabular-overfit/{d} - {i} - {result['tn_mean_score']} - {result['tn_rank']}.txt",
                    "w",
                ) as f:
                    f.write(dumps(result))

    except:
        print("Quitting evaluation early")

    if results:
        df = pd.DataFrame(results)
        df["agents"] = ",".join([a.__name__ for a in agents])
        df.to_csv(f"results/tabular-overfit/tabular-{d}.csv")


if __name__ == "__main__":
    for agents in [
        # [TitForTat, OmegaTFT],
        # [Davis],
        # [Downing],
        # [Feld],
        # [Friedman],
        # [Graaskamp],
        # [Grofman],
        # [Joss],
        # [Nydegger],
        # [RandomAgent],
        # [Shubik],
        # [SteinAndRapoport],
        # [TidemanAndChieruzzi],
        # [Tullock],
        [Borufsen],
        [Champion],
        [Leyvraz],
        [Black],
        [Cave],
        [GraaskampAndKatzen],
        [Harrington],
        [TidemanAndChieruzzi2],
        [Weiner],
        [White],
        [Adams],
    ]:
        main(agents)
