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
#"""
#abstract Regressor
#type LinearRegressor <: Regressor
  #w::Array{Float64}
#end
#predict(r::LinearRegressor,phi) = dot(r.w,phi)
#
#type NeuralNetworkRegressor <: Regressor
  ##probably just hold someone elses implementation parameters
#end
#etc....
#"""

module ReinforcementLearning

export Model
export ActionSpace, domain, DiscreteActionSpace
export BlackBoxModel, init, isterminal, next
export generate_tilecoder, test, bin
export EpsilonGreedyPolicy, SoftmaxPolicy, Policy, DiscretePolicy, weights, action, range
export Solver, Simulator, solve, simulate
export ForgetfulLSTDParam, SARSAParam, TrueOnlineTDParam, LSPIParam
export Minibatcher, NullMinibatcher, UniformMinibatcher
export AnnealerParam, NullAnnealer, MomentumAnnealer, NesterovAnnealer, AdagradAnnealer,AdadeltaAnnealer, AdamAnnealer,RMSPropAnnealer
export ExperienceReplayer, NullExperienceReplayer, UniformExperienceReplayer
export FeatureExpander, ActionFeatureExpander, NullFeatureExpander, iFDDExpander, expand, update

using PyPlot #for solver.grandiloquent
using Interact
import StatsBase: sample, WeightVec #for policy.SoftmaxPolicy
import Base.dot
using HypothesisTests #for utils.test...


#typealias Uses_2nd_A Union{SARSAParam}

typealias RealVector Union{Array{Float64,1},Array{Int,1},SparseMatrixCSC{Float64,Int},SparseMatrixCSC{Int,Int}}
typealias RealMatrix Union{Array{Float64,2},Array{Int,2},SparseMatrixCSC{Float64,Int},SparseMatrixCSC{Int,Int}}
dot(x::Array,y::SparseMatrixCSC) = (x'*y)[1]
dot(x::SparseMatrixCSC,y::Array) = dot(y,x)
dot(x::SparseMatrixCSC,y::SparseMatrixCSC) = (x'*y)[1]

import Base.assert #in order for other asserts to be allowed
function assert(expr,val,fn::Function= ==,varname::AbstractString="")
	if !fn(expr,val)
    error("Assertion failed: $varname : expected $val, got $expr")
	end
end

abstract AnnealerParam
abstract ExperienceReplayer
abstract Minibatcher
abstract UpdaterParam
abstract ActionSpace
abstract Policy
abstract Model
abstract FeatureExpander
abstract ActionFeatureExpander <: FeatureExpander

include("BlackBoxModel.jl")

include("policy.jl")

include("learners.jl")

include("simulator.jl")

include(joinpath("solvers","__solvers.jl"))

#"""
#for solver in filter(isfile,readdir())
#  include(joinpath("solvers",solver))
#end
#"""

include("solve.jl")

include("utils.jl")

end #mdule
