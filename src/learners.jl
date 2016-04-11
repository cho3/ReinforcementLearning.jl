#learners.jl
#this might be broken into learningrate.jl and minibatch.jl and experiencereplay.jl in the future


type GradientClipper
  max_norms::Dict{Int,Float64}
  norm::Real
  relative::Bool #scale the max_norm by vector length
  function GradientClipper(;
                            max_norms::Union{Float64,Array{Float64,1},Dict{Int,Float64}}=0.5, #0.5
                            norm::Real=2.,
                            relative::Bool=true)
    self = new()
    self.norm = norm
    self.relative = relative
    if typeof(max_norms) == Float64
      self.max_norms = Dict{Int,Float64}(1=>max_norms)
    else
      self.max_norms = max_norms
    end

    return self

  end
end

function clip!(gc::GradientClipper,dx::Union{RealMatrix,RealVector},i::Int)
  u = norm(dx,gc.norm)
  max_norm = get(gc.max_norms,i,gc.max_norms[1])
  if gc.relative
    max_norm *= length(dx)^(1./gc.norm)
  end
  if u > max_norm
    dx *= max_norm/u
  end
  return dx
end

include("RateAdapters.jl")

include("Annealers.jl")

######################################

include("ExperienceReplay.jl")

##################################
include("Minibatch.jl")

#TODO: something that halves the learning rate every t0 time steps -- probably not that useful, but whatever

##############################################
include("FeatureExpander.jl")
#=
function action{T}(policy::EpsilonGreedyPolicy,updater::iFDDParam,s::T)
  #darn it thought i could just cast it :(
  r = rand(p.rng)
  if r < p.eps
    return domain(p.A)[rand(p.rng,1:length(p.A))]
  end
  Qs = zeros(length(p.A))
  for (i,a) in enumerate(domain(p.A))
    #NOTE: here, case feature function
    Qs[i] = dot(weights(u),expand(updater,p.feature_function(s,a))) #where is a sensible place to put w?
  end
  return domain(p.A)[indmax(Qs)]
end
=#

#TODO: softmax policy
#TODO: DiscretePolicy

#TODO: solve--expand feaures first




##############################################
