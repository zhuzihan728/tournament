"""
    grid search for high/low performing agents 
    trained by DQN 1 hidden layer model
    
    lookback: number of historical movements
    n1: hidden layer unit number
    epsilon: 
    epsilon_decay
    learning_rate
    discount_rate
"""

import itertools
from datetime import datetime
from json import dumps

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tournament.agents.axelrod_first import (
    Davis,
    Downing,
    Feld,
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
    GraaskampAndKatzen,
    Harrington,
    TidemanAndChieruzzi2,
    Weiner,
    White,
)
from tournament.agents.constant import AllC, AllD
from tournament.agents.pavlov import Pavlov
from tournament.agents.q_learning.dqn import DeepQLearner
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


class QNetwork(nn.Module):
    def __init__(self, lookback, n1=32):
        super().__init__()

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(2 * lookback, n1)
        self.layer2 = nn.Linear(n1, n1)
        self.layer3 = nn.Linear(n1, 2)

        nn.init.kaiming_uniform_(self.layer1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer3.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        x = x.unsqueeze(dim=0)
        x = self.flatten(x)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))

        return x


class DQN(DeepQLearner):
    def __init__(
        self, lookback, n1, epsilon, epsilon_decay, learning_rate, discount_rate
    ):
        super().__init__()

        self.lookback = lookback
        self.epsilon = epsilon

        self._epsilon_decay = epsilon_decay
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate
        self._q_network = QNetwork(self.lookback, n1)


def main():
    agents = agents = [
        TitForTat,
        TidemanAndChieruzzi,
        Borufsen,
        GraaskampAndKatzen,
        Nydegger,
        Grofman,
        Shubik,
        Champion,
        Leyvraz,
    ]

    grid = {
        "lookback": [1, 2, 4, 8],
        "n1": [8, 12, 16, 24],
        "epsilon": [0.05, 0.1, 0.2],
        "epsilon_decay": [0.0],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "discount_rate": [0.95, 0.99],
    }

    d = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    results = []
    try:
        space = list(itertools.product(*grid.values())) # list of all cartesian product tuples of the hyper-parameters
        size = len(space)
        for i, hyperparameters in enumerate(space):
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H-%M-%S')} | {i + 1}/{size}]",
                *hyperparameters,
                sep="\t",
            )
            result, agent = train_and_evaluate(
                agents, DQN, epochs=250, **dict(zip(grid.keys(), hyperparameters))
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
            if result["tn_mean_score"] > 750 or result["tn_rank"] > 27:   # if result is very good, or very bad, save the model as .pt, and as .txt
                torch.save( 
                    agent._q_network.state_dict(),
                    f"models/dqn/{d} - 1hl - {i} - {result['tn_mean_score']} - {result['tn_rank']}.pt",
                )
                with open(
                    f"models/dqn/{d} - 1hl - {i} - {result['tn_mean_score']} - {result['tn_rank']}.txt",
                    "w",
                ) as f:
                    f.write(dumps(result))
    except Exception as e:
        print("Quitting evaluation early:", str(e))
    except:
        print("Quitting evaluation early")

    if results:                                 # save the results to a spread sheet for future comparation
        df = pd.DataFrame(results)
        df["agents"] = ",".join([a.__name__ for a in agents])
        df.to_csv(f"results/dqn/dqn-1hl-{d}.csv")


if __name__ == "__main__":
    main()
