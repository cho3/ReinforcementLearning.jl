#CaveWorld.jl
# A blind gridworld pomdp taken from: http://www.cs.cmu.edu/afs/cs/academic/class/15780-s13/www/hw/hw3_sol.pdf
# this is the localization problem

push!(LOAD_PATH,joinpath("..","src"))
using ReinforcementLearning
import Iterators.product

#Base Model
type CaveWorld <: Model
    W::Int #x
    H::Int #y
    r_correct::Float64
    r_wrong::Float64
    r_observe::Float64
    p_wall::Float64
    p_not_wall::Float64
    invalid_states::Set{Array{Int,1}}
    goal::Array{Int,1}
    function CaveWorld(;
                        W::Int=3,
                        H::Int=3,
                        r_correct::Float64=100.,
                        r_wrong::Float64=-1000.,
                        r_observe::Float64=-10.,
                        p_wall::Float64=0.9,
                        p_not_wall::Float64=0.5,
                        invalid_states::Set{Array{Int,1}}=Set{Array{Int,1}}(),
                        goal::Array{Int,1}=[3,1])
      self = new()
      self.W = W
      self.H = H
      self.r_correct = r_correct
      self.r_wrong = r_wrong
      self.r_observe = r_observe
      self.p_wall = p_wall
      self.p_not_wall = p_not_wall
      if length(invalid_states) == 0 #idk what better way to do this set construction is weird
        push!(invalid_states,[1,1])
        push!(invalid_states,[1,2])
        push!(invalid_states,[2,2])
      end
      self.invalid_states = invalid_states
      self.goal = goal
      return self
    end
end

typealias State Array{Int,1} #(x,y,dir), 0=>down,1=>left,2=>up,3=>right
typealias Action Int #0=>observe, 2=>forward, -1=>left, 1=>right, 4=> announce
typealias Obs Bool #is wall in front

A = DiscreteActionSpace([0;1;-1;2;4])
#init
function init(m::CaveWorld,rng::AbstractRNG)
  _states = product(1:m.W,1:m.H)
  states = [Int[y for y in x] for x in _states]
  for bad_state in m.invalid_states
    #INEFFICIENT XXX
    _ind = find(x->x==bad_state,states)
    if length(_ind) != 0
      splice!(states,_ind[1])
    end
  end
  return states[rand(rng,1:length(states))]
end


#next
function next(rng::AbstractRNG,m::CaveWorld,s::State,a::Action)
  x = s[1]
  y = s[2]
  dir = s[3]
  #turning
  if abs(a) == 1
    dir_ = mod(dir+a,4)
    return [x;y;dir_]
  elseif a == 2
    dir2xy = Dict{Int,Array{Float64,1}}(0=>[-1;0],1=>[0;-1],2=>[1;0],3=>[0;1])
    dx,dy = dir2xy[dir]
    x_ = max(min(x+dx,m.W),1)
    y_ = max(min(y+dy,m.H),1)
    if [x_;y_] in m.invalid_states
      return [x;y;dir]
    end
    return [x_;y_;dir]
  end
  #if announce or observe
  return [x;y;dir]
end

#it never ends :(
isterminal(rng::AbstractRNG,m::CaveWorld,s::State,a::Action) = a == 4

#Reward
function reward(rng::AbstractRNG,m::CaveWorld,s::State,a::Action)
  if (a == 4)
    if (s == m.goal)
      return m.r_correct
    else
      return m.r_wrong
    end
  elseif a == 0
    return m.r_observe
  end
  return 0.
end


function observe(rng::AbstractRNG,m::CaveWorld,s::State,a::Action)
  #do a pseudo step--if can't, then observe a wall
  dir2xy = Dict{Int,Array{Float64,1}}(0=>[-1;0],1=>[0;-1],2=>[1;0],3=>[0;1])
  dx,dy = dir2xy[dir]
  #check edges
  if max(min(x+dx,m.W),1) != x+dx
    return rand(rng) < m.p_wall
  elseif max(min(y+dy,m.W),1) != y+dy
    return rand(rng) < m.p_wall
  end
  #check invalid states
  x_ = max(min(x+dx,m.W),1)
  y_ = max(min(y+dy,m.H),1)
  if [x_;y_] in m.invalid_states
    return rand(rng) < m.p_wall
  end
  #no wall!
  return rand(rng) < m.p_not_wall
end

observe(rng::AbstractRNG,m::CaveWorld,s::State) = false

bbm = BlackBoxModel(CaveWorld(),init,next,reward,isterminal,observe=observe)

feature_function(o::Obs) = o ? [1.;0.]: [0.;1.]

#heuristics
#Random
#never announce
#always announce

#=
#stuff for actor-critic
import Base.convert
import Base.vec

vec(x::Float64) = Float64[x]

Float64(x::Array{Float64,1}) = x[1]
=#
