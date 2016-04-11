#MPC.jl
#note: requires NLOpt Package

#workaround for parallel NLOpt: credit to scidom/ChaosCommunications (github)
abstract Optimizer
typealias VF64OrVoid Union{Vector{Float64}, Void}
typealias F64OrVoid Union{Float64, Void}
# NLoptimizer holds the parameters passed to Opt. This is a workaround for the lack of serialization of Opt.
immutable NLoptimizer <: Optimizer
  npars::Int # Number of optimization parameters
  alg::Symbol # Optimization algorithm to use
  lb::VF64OrVoid # Lower bound constraint
  ub::VF64OrVoid # Upper bound constraint
  absftol::Float64 # Stopping criterion: absolute tolerance on function value
  relftol::Float64 # Stopping criterion: relative tolerance on function value
  absxtol::Float64 # Stopping criterion: absolute tolerance on optimization parameters
  relxtol::Float64 # Stopping criterion: relative tolerance on optimization parameters
  nfeval::Int64 # Stopping criterion: maximum number of function evaluations
  maxtime::Float64 # Stopping criterion: maximum optimization time

  function NLoptimizer(npars::Int, alg::Symbol, lb::VF64OrVoid, ub::VF64OrVoid, absftol::Float64,
    relftol::Float64, absxtol::Float64, relxtol::Float64, nfeval::Int64, maxtime::Float64)
    if isnan(absftol) && isnan(relftol) && isnan(absxtol) && isnan(relxtol) && nfeval < 0 && maxtime < 0
      error("A stopping criterion must be specified.")
    end

    new(npars, alg, lb, ub, absftol, relftol, absxtol, relxtol, nfeval, maxtime)
  end
end

NLoptimizer(npars::Int;
  alg::Symbol=:LN_COBYLA,
  lb::VF64OrVoid=Void,
  ub::VF64OrVoid=Void,
  absftol::Float64=1e-32,
  relftol::Float64=NaN,
  absxtol::Float64=1e-32,
  relxtol::Float64=NaN,
  nfeval::Int64=1_000,
  maxtime::Float64=-1.0) =
  NLoptimizer(npars, alg, lb, ub, absftol, relftol, absxtol, relxtol, nfeval, maxtime)

function NLoptimizer(;
  alg::Symbol=:LN_COBYLA,
  lb::F64OrVoid=Void,
  ub::F64OrVoid=Void,
  absftol::Float64=1e-32,
  relftol::Float64=NaN,
  absxtol::Float64=1e-32,
  relxtol::Float64=NaN,
  nfeval::Int64=1_000,
  maxtime::Float64=-1.0)
  lbvec = lb == Void ? Void : [lb]
  ubvec = ub == Void ? Void : [ub]
  NLoptimizer(1, alg, lbvec, ubvec, absftol, relftol, absxtol, relxtol, nfeval, maxtime)
end

function convert(::Type{Opt}, o::NLoptimizer)
  opt = Opt(o.alg, o.npars)

  if o.lb != Void; lower_bounds!(opt, o.lb); end
  if o.ub != Void; upper_bounds!(opt, o.ub); end
  if !isnan(o.absftol) ftol_abs!(opt, o.absftol); end
  if !isnan(o.relftol) ftol_rel!(opt, o.relftol); end
  if !isnan(o.absxtol) xtol_abs!(opt, o.absxtol); end
  if !isnan(o.relxtol) xtol_rel!(opt, o.relxtol); end
  if !(o.nfeval < 0) maxeval!(opt, o.nfeval); end
  if !(o.maxtime < 0) maxtime!(opt, o.maxtime); end

  return opt
end


###########################

#NOTE: use bin(x,lb,ub,nb_bins) for action_to_vector

type MPCPolicy <: Policy
  time_horizon::Int #necessary? encapsulated by length of x vectors?
  nb_var_per_action::Int
  compute_time::Float64
  vector_to_action::Function
  optimizer::NLoptimizer#Opt #use NLOpt symbol? assume [-1,1] bound
  x_warm::Array{Array{Float64,1},1}
  bbm::BlackBoxModel
  init_map::Dict{Int,AbstractString}
  verbose::Bool
  function MPCPolicy(bbm::BlackBoxModel,
                      vector_to_action::Function,
                      n::Int;
                      nb_beams::Int=3,
                      compute_time::Float64=10.,
                      time_horizon::Int=5,
                      verbose::Bool=false,
                      lb::AbstractVector=zeros(n*time_horizon),
                      ub::AbstractVector=ones(n*time_horizon),
                      optimizer::Symbol=:LN_BOBYQA) #default to most modern derivative-free local optim
    self = new()
    self.compute_time = compute_time
    self.nb_var_per_action=n
    self.time_horizon = time_horizon #necessary?
    self.vector_to_action = vector_to_action
    self.time_horizon = time_horizon #necesary?
    #n is the number of variables to define an action
    #=
    optimizer = Opt(optimizer,n*time_horizon)
    maxtime!(optimizer,compute_time/nb_beams)
    lower_bounds!(optimizer,lb)
    upper_bounds!(optimizer,ub)
    =#
    optimizer = NLoptimizer(n*time_horizon,alg=optimizer,lb=lb,ub=ub,maxtime=compute_time/nb_beams)
    self.optimizer = optimizer
    #init_map: 1=rand/0.5, 2=ones,zeros, 3=ones,zeros,0.5, 4+=ones,zeros,0.5,rand...
    if nb_beams == 1
      init_map = Dict{Int,AbstractString}(1=>"rand")
    else
      init_map = Dict{Int,AbstractString}(1=>"one",2=>"zero")
      for ind in 3:nb_beams
        init_map[ind] = "rand"
      end
    end
    self.init_map = init_map
    self.x_warm = [init_weights(n*time_horizon,init_map[i]) for i = 1:nb_beams]
    self.bbm = deepcopy(bbm)
    self.verbose = verbose

    return self
  end
end

function lookahead{T}(p::MPCPolicy,s::T,x::Vector{Float64})
  p.bbm.state = deepcopy(s) #necessary?
  #println(p.bbm.state)
  as = p.vector_to_action(x)
  R = 0.
  for a in as
    #println(a)
    if isterminal(p.bbm,a) #not sure if right--does it capture failure/success conditions?
      r,o = next(p.bbm,a) #again, not sure if right
      R += r
      break
    end
    r,o = next(p.bbm,a)
    #println(r)
    R += r
  end
  return R
end

function rollforward(p::MPCPolicy,x::Vector{Float64},i::Int)
  x = circshift(x,-p.nb_var_per_action) #circhshift PUSHES back
  x[end-p.nb_var_per_action+1:end] = init_weights(p.nb_var_per_action,p.init_map[i]) #zeros, ones, 0.5, rand
  return x
end

#NOTE: check if i need to do some deepcopy magic with s
function action{T}(p::MPCPolicy,s::T)
  #push x_warm forward, random initialization for last
  #set up optimization problem
  f_opt(x::Vector{Float64},grad::Union{Vector,Matrix}) = lookahead(p,s,x)
  #optimize!
  #println(f_opt)
  opt = convert(Opt,p.optimizer)
  max_objective!(opt,f_opt)
  vals = zeros(length(p.x_warm))
  for (i,x0) in enumerate(p.x_warm)
    x0 = rollforward(p,x0,i)
    maxf,maxx,ret = optimize(opt,x0)
    vals[i] = maxf
    p.x_warm[i] = maxx
  end
  #select best
  idx_max = indmax(vals)
  if p.verbose
    println(idx_max)
    println(vals)
    println(p.vector_to_action(p.x_warm[idx_max])[1])
  end
  #convert to action, pop off top
  #return action
  return p.vector_to_action(p.x_warm[idx_max])[1]
end
