#Tiger.jl
# An implementation of the tiger pomdp consistent with my BBM type

push!(LOAD_PATH,joinpath("..","src"))
using ReinforcementLearning

#Base Model
type TigerPOMDP <: Model
    r_listen::Float64
    r_correct::Float64
    r_wrong::Float64
    p_left_correct::Float64
    p_right_correct::Float64
    p_tiger_left::Float64
    function TigerPOMDP(;
                        r_listen::Float64=-1.,
                        r_correct::Float64=10.,
                        r_wrong::Float64=-100.,
                        p_left_correct::Float64=0.85,
                        p_right_correct::Float64=0.85,
                        p_tiger_left::Float64=0.5)
      self = new()
      self.r_listen = r_listen
      self.r_correct = r_correct
      self.r_wrong = r_wrong
      self.p_left_correct = p_left_correct
      self.p_right_correct = p_right_correct
      self.p_tiger_left = p_tiger_left
      return self
    end
end

typealias State Bool #is tiger left?
typealias Action Int #0=>listen, 1=>openleft, 2=>openright
typealias Obs Int #0=>Null, 1=>left, 2=>right
#init
init(m::TigerPOMDP,rng::AbstractRNG) = rand(rng) < m.p_tiger_left


#next
function next(rng::AbstractRNG,m::TigerPOMDP,s::State,a::Action)
  if a != 0
    return init(m,rng)
  end
  return s
end

isterminal(rng::AbstractRNG,m::TigerPOMDP,s::State,a::Action) = (s && (a == 1)) || (!s && (a == 2))
isterminal2(rng::AbstractRNG,m::TigerPOMDP,s::State,a::Action) = (a != 0)


#Reward
function reward(rng::AbstractRNG,m::TigerPOMDP,s::State,a::Action)
  if a == 0
    return m.r_listen
  end
  return isterminal(rng,m,s,a)? m.r_wrong: m.r_correct
end

function reward2(rng::AbstractRNG,m::TigerPOMDP,s::State,a::Action)
  if a == 0
    return m.r_listen
  end
  if s
    if a == 1
      return m.r_wrong
    else
      return m.r_correct
    end
  else #! left = right
    if a == 2
      return m.r_wrong
    else
      return m.r_correct
    end
  end
end

function observe(rng::AbstractRNG,m::TigerPOMDP,s::State,a::Action)
  if a != 0
    return 0
  end
  if s
    return rand(rng) < m.p_left_correct ? 1 : 2
  end
  return rand(rng) < m.p_right_correct ? 2 : 1
end

observe(rng::AbstractRNG,m::TigerPOMDP,s::State) = 0 #Null observation

bbm = BlackBoxModel(TigerPOMDP(),init,next,reward,isterminal,observe=observe)
bbm2 = BlackBoxModel(TigerPOMDP(),init,next,reward2,isterminal2,observe=observe)

feature_function(o::Obs) = sparsevec([o+1],[1.],3)

A = DiscreteActionSpace([0;1;2])

#HEURISITCS
type Left <: Policy
  exp::TrueNullFeatureExpander
end
ReinforcementLearning.action(::Left,::Obs) = 1

type Right <: Policy
  exp::TrueNullFeatureExpander
end
ReinforcementLearning.action(::Right,::Obs) = 2

type Random <: Policy
  exp::TrueNullFeatureExpander
end
ReinforcementLearning.action(::Random,::Obs) = rand(0:2)

type ListenOnce <: Policy
  exp::TrueNullFeatureExpander
end
function ReinforcementLearning.action(::ListenOnce,o::Obs)
  if o == 0
    return o
  end
  return o == 1 ? 2: 1
end

type ListenOnly <: Policy
  exp::TrueNullFeatureExpander
end
ReinforcementLearning.action(::ListenOnly,::Obs) = 0

type ListenK <: Policy
  exp::TrueNullFeatureExpander
  counts::Array{Int,1}
  k::Int
end
function ReinforcementLearning.action(p::ListenK,o::Obs)
  if o == 0
    p.counts = [0;0]
    return 0 #listen
  end
  p.counts[o] += 1
  if abs(p.counts[1]-p.counts[2]) >= p.k
    return p.counts[1] > p.counts[2] ? 2 : 1
  end
  return 0
end

ListenK(k::Int) = ListenK(TrueNullFeatureExpander(),[0;0],k)
ListenK() = ListenK(2)
Left() = Left(TrueNullFeatureExpander())
Right() = Right(TrueNullFeatureExpander())
Random() = Random(TrueNullFeatureExpander())
ListenOnce() = ListenOnce(TrueNullFeatureExpander())
ListenOnly() = ListenOnly(TrueNullFeatureExpander())
#=
#stuff for actor-critic
import Base.convert
import Base.vec

vec(x::Float64) = Float64[x]

Float64(x::Array{Float64,1}) = x[1]
=#
