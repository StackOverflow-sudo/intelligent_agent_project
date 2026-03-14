# COMP6203 -- Maritime Shipping Agent

## Overview

This project implements an autonomous trading agent for the **MABLE
maritime shipping simulator**, developed as part of the **COMP6203
Intelligent Agents coursework** at the University of Southampton.

In this environment, shipping companies compete in **reverse
second‑price auctions** for transportation contracts. Each agent must
decide: - Which trades to bid on - What price to bid - How to schedule
vessels to execute awarded contracts

The key challenge is that **bidding and scheduling decisions are
interdependent**: the cost of transporting a request depends on the
current fleet schedule, vessel locations, and time-window constraints.

------------------------------------------------------------------------

## Environment

The system uses the **MABLE simulator**, an event-based maritime
logistics environment where:

-   Cargo transportation opportunities appear periodically
-   Companies submit bids for trades
-   The lowest bidder wins but receives the **second-lowest bid
    payment**
-   Winning companies must transport the cargo using their fleet

Ships must satisfy: - Cargo capacity constraints - Pickup and drop-off
time windows - Fuel consumption costs

------------------------------------------------------------------------

## Key Features of Our Agent

Our agent integrates the following components:

1.  **Schedule‑Aware Cost Estimation**
2.  **Adaptive Bidding Strategy**
3.  **Online Market Learning**
4.  **Profit‑Driven Post‑Auction Scheduling**

------------------------------------------------------------------------

## Agent Decision Workflow

For each auction round the agent performs:

1.  **Cost estimation** for incoming transportation requests
2.  **Bid generation** based on estimated execution costs
3.  **Market learning** from auction outcomes
4.  **Schedule updates** for newly awarded contracts

This design keeps bidding decisions aligned with scheduling feasibility.

------------------------------------------------------------------------

## Cost Estimation

Instead of computing an optimal schedule (which is computationally
expensive), the agent uses a **multi‑start insertion heuristic**:

-   Multiple candidate schedules are generated using random insertion
    orders
-   Feasible schedules are evaluated
-   The **Top‑K schedules** are selected
-   Marginal insertion costs are estimated from these schedules

This approach produces stable cost estimates while remaining
computationally efficient.

Outputs: - Marginal insertion cost m(r) - Schedule stability score p(r)

------------------------------------------------------------------------

## Bidding Strategy

Bids combine: - Direct transportation cost proxy - Marginal insertion
cost - Market competitiveness signal

A **regime‑dependent pricing strategy** is used:

  Market Condition   Strategy
  ------------------ -------------------
  Tight market       Conservative bids
  Medium market      Balanced bids
  Loose market       Aggressive bids

Market strength is estimated using an **EWMA price signal** that
dynamically adjusts bidding aggressiveness.

------------------------------------------------------------------------

## Market Learning

Because competitors' bids are only partially observable, the agent
learns from auction outcomes using:

-   Price‑to‑cost ratio tracking
-   Exponentially weighted moving averages (EWMA)
-   Left‑censored learning from partial bid observations

These updates allow the agent to adapt its bidding strategy over time.

------------------------------------------------------------------------

## Post‑Auction Scheduling

When contracts are awarded, the agent integrates them into fleet
schedules using **profit‑driven insertion**.

A request is accepted only if:

payment − additional_cost ≥ 0

This avoids infeasible or loss‑making transportation plans.

------------------------------------------------------------------------

## Experimental Evaluation

The agent was evaluated against several baseline competitors:

-   **MyArchEnemy** -- fixed profit margin agent
-   **TheScheduler** -- heuristic scheduling agent
-   **groupn1** -- adaptive profit agent

Experiments varied: - Market competitiveness - Fleet size

Results indicate: - Stable performance across environments - Increasing
income in less competitive markets - Low penalty rates due to reliable
scheduling feasibility

------------------------------------------------------------------------

## Project Structure

Example project structure:

    project/
    │
    ├── groupn.py(agent source code)
    │
    ├── main_playground.py
    │
    ├── report_group1.pdf
    ├── README.md

------------------------------------------------------------------------

## Installation

Install the MABLE simulator:

    pip install mable

Ensure the file **mable_resources.zip** is located in the project
directory.

------------------------------------------------------------------------

## Running the Simulation

Run the simulation using:

    python run_simulation.py

After execution a metrics file will be generated:

    metrics_competition_<id>.json

To view summary results:

    mable overview metrics_competition_<id>.json

------------------------------------------------------------------------

## Limitations

Current limitations include:

-   Scheduling heuristics may degrade with large fleets
-   Market learning may react slowly to sudden strategy shifts
-   Early auction rounds may be overly conservative due to limited data

Future improvements could include: - Opponent modelling - Reinforcement
learning for bidding - More advanced scheduling heuristics

------------------------------------------------------------------------

## Authors

COMP6203 Intelligent Agents -- Group 1

University of Southampton

Code writting: Ruiqi Wang
draft report: Ruiqi Wang
final report: Group 1
