# External substrates (not redistributed)

MtM-Bench scores contestants on two external substrates whose gold belongs to their authors.
Download them yourself; the loaders point at your clone.

## AgentProcessBench (process-quality + APB leaderboard)
    git clone https://github.com/RUCBM/AgentProcessBench ~/agentprocessbench
    mtm-bench apb-leaderboard --apb-dir ~/agentprocessbench

## AgentErrorBench (attribution axis, via AgentDebug)
    See https://arxiv.org/abs/2509.25370 for the release location (Google Drive);
    the `aeb_gold` adapter consumes its consensus labels + ALFWorld trajectories.
