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
export EpsilonGreedyPolicy, SoftmaxPolicy, Policy, DiscretePolicy, weights, action, range, length
export Solver, Simulator, solve, simulate
export ForgetfulLSTDParam, SARSAParam, TrueOnlineTDParam, LSPIParam, QParam, GQParam
export MPCPolicy
export Minibatcher, NullMinibatcher, UniformMinibatcher
export AnnealerParam, NullAnnealer, MomentumAnnealer, NesterovAnnealer, AdagradAnnealer,AdadeltaAnnealer, AdamAnnealer,RMSPropAnnealer
export ExperienceReplayer, NullExperienceReplayer, UniformExperienceReplayer
export FeatureExpander, ActionFeatureExpander, NullFeatureExpander, iFDDExpander,iFDDProperExpander, expand, update!, pad!, expand2
export generate_tilecoder, test, bin, generate_radial_basis, sample, powerset, sortedpowerset
export save, load, load_policy

using PyPlot #for solver.grandiloquent
using Interact
import Base: dot, length, values
import StatsBase: sample, WeightVec,values #for policy.SoftmaxPolicy
using HypothesisTests #for utils.test...
using JLD #for saving/loading/model persistence
using NLopt #for MPC


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

############################################################################
#Saving/Loading/Model Persistence
#JLD can't support generic functions
#=
  BlackBoxModel can't be saved.
  Policy: feature_function can't be saved
  FeatureExpander: can be saved
  Annealer: can be saved
  ExperienceReplayer: can be saved
  Simulator: can be saved
  UpdaterParam: can be saved
  SolverHistory: can be saved
  Solver: can be saved
=#

name_type_dict = Dict{DataType,AbstractString}(UpdaterParam=>"updater",
                                              Solver=>"solver",
                                              Simulator=>"simulator",
                                              FeatureExpander=>"featureexpander",
                                              AnnealerParam=>"annealer",
                                              ExperienceReplayer=>"experiencereplayer",
                                              SolverHistory=>"solverhistory")
function save(x...;fname::AbstractString="rl_save.jld",verbose::Bool=true)
  if splitext(fname)[2] != ".jld"
    warning("Writing to file: $fname with improper extension!")
  end
  jldopen(fname,"w") do file
    addrequire(file,ReinforcementLearning)
    for var in x
      if typeof(var) in keys(name_type_dict)
        write(file,name_type_dict[typeof(var)],var)
      elseif typeof(var) == Policy
        write(file,"policy",string(typeof(var)))
        save(var,file)
      end
    end
  end
  if verbose
    println("Wrote policy to \"$fname\"!")
  end
  return fname
end

function load(fname::AbstractString;verbose::Bool=true)
  #check existence
  data = JLD.load(fname)
  if "policy" in keys(data)
    println("Call `load_policy(file_name,feature_function)` to load the policy!")
    #load_policy!(data["policy"],data)
  end
  return data
end

#NOTE: this probably doesn't work--exporting not supported
function load!{T}(x::T,fname::AbstractString;verbose::Bool=true)
  #check existence
  if !(T in keys(name_type_dict))
    error("Loading of type: \"$T\" via load!() is not supported!")
  end
  c = jldopen(fname, "r") do file
    t = read(file, name_type_dict[T])
  end
  return t
end

#No JldFile type :(
#TODO: remove this?
save(x::Policy,file) = save(x,file) #see policy.jl for specific methods

#=
Can potentially do more fine-grained loading, i.e.:
  load!(p::Policy;fname=...) = read(file,"policy)")
=#

end #mdule
