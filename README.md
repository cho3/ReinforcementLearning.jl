# Reinforcement Learning.jl

##Installation
`cd /path/to/install

git clone https://www.github.com/cho3/ReinforcementLearning.jl.git`

##Usage
`include(joinpath("path","to","install","ReinforcementLearning.jl"))

using ReinforcementLearning

bbm = BlackBoxModel()

policy = EpsilonGreedyPolicy()

updater = SARSAParam()

solver = Solver()

trained_policy = solve(updater,bbm,policy)

sim = Simulator()

simulate(sim,bbm,trained_policy)`
