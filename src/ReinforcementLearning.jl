#ReinforcementLearning.jl
#desc: The author's attempt at a semi-consistent reinforcement learning package that implements a few different algorithms
#auth: Christopher Ho
#affil: Stanford University
#date: 1/11/2016

#TODO: future file demarcations
"""
ReinforcementLearning.jl
BlackBoxModel.jl
simulator.jl
solver.jl
annealer.jl
policy.jl
minibatch.jl
experiencereplay.jl
solvers/
  ForgetfulLSTD.jl
  SARSA.jl #also q-learning, also GQ variations
  DoubleQ.jl
  DeterministicPolicyGradient.jl
  LSPolicyIteration.jl
"""

#TODO: can probably add an additional layer of abstraction:
"""
abstract Regressor
type LinearRegressor <: Regressor
  w::Array{Float64}
end
predict(r::LinearRegressor,phi) = dot(r.w,phi)

type NeuralNetworkRegressor <: Regressor
  #probably just hold someone elses implementation parameters
end
#etc....
"""
#TODO: turn into a module....

include("BlackBoxModel.jl")
include("policy.jl")
include("simulator.jl")
include("learners.jl")
include(joinpath("solvers","__solvers.jl"))
"""
for solver in filter(isfile,readdir())
  include(joinpath("solvers",solver))
end
"""
include("solve.jl")
