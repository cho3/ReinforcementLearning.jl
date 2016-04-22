#GridWorld.jl

using PyPlot
using Interact

typealias State Tuple{Int,Int}
typealias Action Tuple{Int,Int}

type GridWorldModel <: Model
  W::Int
  H::Int
  p_other::Float64
  reward_locs::Dict{State,Float64}
  collide_cost::Float64
  A::Array{Action,1}
end

A = Action[(0,0),(1,0),(-1,0),(0,1),(0,-1)]

init2(m::GridWorldModel,rng::AbstractRNG) = (rand(rng,1:m.W),rand(rng,1:m.H))
init1(m::GridWorldModel,rng::AbstractRNG) = (1,1)

isend1(rng::AbstractRNG,m::GridWorldModel,s::State,a::Action) = s == (m.W,m.H)
isend2(rng::AbstractRNG,m::GridWorldModel,s::State,a::Action) = false

function reward(rng::AbstractRNG,m::GridWorldModel,s::State,a::Action)
  x_ = s[1] + a[1]
  y_ = s[2] + a[2]

  if (x_ < 1) || (x_ > m.W)
    return m.collide_cost
  elseif (y_ < 1) || (x_ > m.H)
    return m.collide_cost
  end

  x_ = max(min(x_,m.W),1)
  y_ = max(min(y_,m.H),1)

  return get(m.reward_locs,(x_,y_),0.)

end

function next(rng::AbstractRNG,m::GridWorldModel,s::State,a::Action)
  A_other = setdiff(A,a)

  if rand(rng) < m.p_other
    _a = A_other[rand(rng,1:length(A_other))]
  else
    _a = a
  end
  x_ = s[1] + _a[1]
  y_ = s[2] + _a[2]

  x_ = max(min(x_,m.W),1)
  y_ = max(min(y_,m.H),1)

  return (x_,y_)
end

function generate_featurefunction(m::GridWorldModel,A::Array{Action,1})

  nb_feat = m.W*m.H*length(A)
  A_indices = [a=>i for (i,a) in enumerate(A)]
  function feature_function(s::State,a::Action)
    active_indices = [s[1]+m.W*(s[2]-1)+m.W*m.H*(A_indices[a]-1)]
    phi = sparsevec(active_indices,ones(length(active_indices)),nb_feat)
    return phi
  end

  return feature_function

end

function visualize(m::GridWorldModel,s::State,a::Action)
  #base grid
  for i = 1:m.W
    for j = 1:m.H
      val = get(m.reward_locs,(i,j),0.)
      if val > 0
        color = "#31B404"
      elseif val < 0
        color = "#FF0000"
      else
        color = "#A4A4A4"
      end
      fill([i;i+1;i+1;i],[j;j;j+1;j+1],color=color,edgecolor="#FFFFFF")
    end #j
  end #i

  #draw agent
  agent_color = "#0101DF"
  x = s[1] + 0.5
  y = s[2] + 0.5
  fill([x-0.5;x;x+0.5;x],[y;y-0.5;y;y+0.5],color=agent_color)
  #draw direction
  arrow(x,y,a[1],a[2],width=0.1,head_width=0.2,head_length=0.5)

end

function visualize(m::GridWorldModel,S::Array{State,1},A::Array{Action,1})
  assert(length(S) == length(A))
  f = figure()
  @manipulate for i = 1:length(S); withfig(f) do
    visualize(m,S[i],A[i]) end
  end
end
