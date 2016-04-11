#CryingBaby.jl
# Crying Baby POMDP
# Copied shamelessly from Mykel, Zach, and Max and JuliaPOMDP/POMDPModels.jl

push!(LOAD_PATH,joinpath("..","src"))
using ReinforcementLearning

#Base Model
type BabyPOMDP <: Model
    r_feed::Float64
    r_hungry::Float64
    p_become_hungry::Float64
    p_cry_when_hungry::Float64
    p_cry_when_not_hungry::Float64
    function BabyPOMDP(;
                        r_feed::Float64=-5.,
                        r_hungry::Float64=-10.,
                        p_become_hungry::Float64=0.1,
                        p_cry_when_hungry::Float64=0.8,
                        p_cry_when_not_hungry::Float64=0.1)
      self = new()
      self.r_feed = r_feed
      self.r_hungry = r_hungry
      self.p_become_hungry = p_become_hungry
      self.p_cry_when_hungry = p_cry_when_hungry
      self.p_cry_when_not_hungry = p_cry_when_not_hungry
      return self
    end
end

typealias State Bool #is hungry
typealias Action Bool #is feeding
typealias Obs Bool #is crying

A = DiscreteActionSpace([true;false])
#init
init(m::BabyPOMDP,rng::AbstractRNG) = rand(rng) < 0.5 #TODO


#next
function next(rng::AbstractRNG,m::BabyPOMDP,hungry::State,feeding::Action)
  if !feeding && hungry
    return hungry
  elseif feeding
    return !hungry #false
  else #!hungry && !feeding
    return rand(rng) < m.p_become_hungry
  end
end

#it never ends :(
isterminal(rng::AbstractRNG,m::BabyPOMDP,s::State,a::Action) = false

#Reward
function reward(rng::AbstractRNG,m::BabyPOMDP,hungry::State,feeding::Action)
  r = 0.0
  if hungry
    r += m.r_hungry
  end
  if feeding
    r += m.r_feed
  end
  return r
end


function observe(rng::AbstractRNG,m::BabyPOMDP,hungry::State,feeding::Action=false)
  if hungry
    return rand(rng) < m.p_cry_when_hungry
  end
  return rand(rng) < m.p_cry_when_not_hungry
end

bbm = BlackBoxModel(BabyPOMDP(),init,next,reward,isterminal,observe=observe)

feature_function(o::Obs) = o ? [1.;0.]: [0.;1.]

#heuristics
type Starve <: Policy
  exp::TrueNullFeatureExpander
end
ReinforcementLearning.action(::Starve,o::Obs) = false #don't feed

type Stuff <: Policy
  exp::TrueNullFeatureExpander
end
ReinforcementLearning.action(::Stuff,o::Obs) = true #always feed

type FeedWhenCrying <: Policy
  exp::TrueNullFeatureExpander
end
ReinforcementLearning.action(::FeedWhenCrying,crying::Obs) = crying

type Random <: Policy
  exp::TrueNullFeatureExpander
end
ReinforcementLearning.action(::Random,crying::Obs) = rand(Bool)

Starve() = Starve(TrueNullFeatureExpander())
Stuff() = Stuff(TrueNullFeatureExpander())
FeedWhenCrying() = FeedWhenCrying(TrueNullFeatureExpander())
Random() = Random(TrueNullFeatureExpander())
#=
#stuff for actor-critic
import Base.convert
import Base.vec

vec(x::Float64) = Float64[x]

Float64(x::Array{Float64,1}) = x[1]
=#
