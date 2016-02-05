#learners.jl
#this might be broken into learningrate.jl and minibatch.jl and experiencereplay.jl in the future

#Vanilla annealer
type NullAnnealer <: AnnealerParam end
AnnealerParam() = NullAnnealer()
anneal!(::AnnealerParam,dw::RealVector,lr::Float64) = lr*dw

#TODO:
"""
exponential decay: lr = lr0*expt(-k*t)
step decay every t0 epoch if mod(t,t0) == 0: lr /= k (or times)
1/t decay: lr = lr0/(1+kt)
"""

#NOTE:"Taken from cs231n.github.io"
#TODO: make sure this doesn't break from minibatcher returning zeros()
#Momentum Update
type MomentumAnnealer <:AnnealerParam
  v::RealVector #having just  single array may be insufficient for more complex things
  mu::Float64 #[0.5, 0.9, 0.95, 0.99]
end
function anneal!(an::MomentumAnnealer,dw::RealVector,lr::Float64)
  an.v = an.mu*an.v - lr*dw #TODO: how to handle lr btwn stuff
  return an.v
end

#Nesterov update
type NesterovAnnealer <: AnnealerParam
  v::RealVector
  mu::Float64
end
function anneal!(an::NesterovAnnealer, dw::RealVector,lr::Float64)
  v_prev = an.v
  v = an.mu*an.v - lr*dw #TODO: how to handle lr btwn tuff
  return -an.mu*v_prev + (1 + an.mu)*v
end

#Adagrad update
type AdagradAnnealer <: AnnealerParam
  cache::RealVector
  fuzz::Float64 #1e-8
end
function anneal!(an::AdagradAnnealer,dw::RealVector,lr::Float64)
  an.cache += dw.^2
  return lr*dw./sqrt(an.cache + an.fuzz)
end

#Adadelta update
type AdadeltaAnnealer <: AnnealerParam
  mu::Float64
  fuzz::Float64
  dw2::RealVector
  dx2::RealVector
end
function anneal!(an::AdadeltaAnnealer,dw::RealVector,lr::Float64)
  an.dw2 = an.mu*an.dw2 + (1-an.mu)*(dw.^2)
  dx = dw.*sqrt(an.dx2 + an.fuzz)./sqrt(an.dw2 + an.fuzz)
  an.dx2 = an.mu*an.dx2 + (1.-an.mu)*(dx.^2)
  return lr*dx
end

#RMSProp update
type RMSPropAnnealer <: AnnealerParam
  cache::RealVector
  fuzz::Float64 #1e-8
  decay_rate::Float64 #[0.9; 0.99; 0.999]
end
function anneal!(an::RMSPropAnnealer,dw::RealVector,lr::Float64)
  an.cache = an.decay_rate*an.cache + (1.-an.decay_rate)*(dw.^2)
  return lr*dw./sqrt(an.cache + an.fuzz)
end

#ADAM update
type AdamAnnealer <: AnnealerParam
  mu::Float64 #0 <= 0.9 < 1
  nu::Float64 #might be bad naming convention, 0<0.999 <1
  u::RealVector
  v::RealVector
  fuzz::Float64 #1e-8
  t::Int #init 0
end
function anneal!(an::AdamAnnealer,dw::RealVector,lr::Float64)
  an.t += 1
  an.v = an.mu*an.v+ (1.-an.mu)*dw
  an.u = an.nu*an.u + (1.-an.nu)*(dw.^2)
  v_ = an.v./(1-an.mu^an.t)
  u_ = an.u./(1.-an.nu^an.t)
  return lr*v_./(sqrt(u_) + an.fuzz)
end

######################################

type NullExperienceReplayer <:ExperienceReplayer end
replay!(er::NullExperienceReplayer,phi,a,r,phi_,a_) = phi,a,r,phi_,a_#placeholder

#TODO: parameterize this?
#TODO: or make it just a pair of feature vectors + reward
type Experience{T}
  phi::RealVector
  a::T
  r::Float64
  phi_::RealVector
  a_::T
end
remember(e::Experience) = e.phi,e.a,e.r,e.phi_,e.a_
type UniformExperienceReplayer <: ExperienceReplayer
  memory::Array{Experience,1}
  nb_mem::Int
  rng::AbstractRNG
  function UniformExperienceReplayer(nb_mem::Int;rng::AbstractRNG=MersenneTwister(21321))
    self = new()
    self.nb_mem = nb_mem
    self.rng = rng
    self.memory = Experience[]
    return self
  end
end
function replay!{T}(er::UniformExperienceReplayer,
                  phi::RealVector,
                  a::T,
                  r::Float64,
                  phi_::RealVector,
                  a_::T)
  e = Experience(phi,a,r,phi_,a_)
  if length(er.memory) < er.nb_mem
    push!(er.memory,e)
  else
    ind = rand(er.rng,1:er.nb_mem)
    #is it more memory efficient to delete somehow first?
    er.memory[ind] = e
  end
  ind = rand(er.rng,1:length(er.memory))
  return remember(er.memory[ind])
end


##################################
type NullMinibatcher <: Minibatcher end
Minibatcher() = NullMinibatcher()
minibatch!(mb::NullMinibatcher,dw::RealVector) = dw

"This is uniform because you could presumably do other weird averaging things"
type UniformMinibatcher <: Minibatcher
  minibatch_size::Int
  dw::Array{Float64,1}
  current_minibatch_size::Int
end
function minibatch!(mb::UniformMinibatcher,dw::RealVector)
  if mb.current_minibatch_size < mb.minibatch_size
    mb.dw += dw
    mb.current_minibatch_size += 1
    return zeros(size(dw))
  else
    dw_ = (mb.dw + dw)./mb.minibatch_size
    mb.current_minibatch_size = 0
    mb.dw = zeros(size(dw))
    return dw_
  end
end

#TODO: something that halves the learning rate every t0 time steps -- probably not that useful, but whatever

##############################################

#iFDD

expand{T}(exp::FeatureExpander,phi::RealVector,a::T) = phi
expand(exp::FeatureExpander,phi::RealVector) = phi
update!(::FeatureExpander,phi::RealVector,del::Float64) = 0

type NullFeatureExpander{T} <: ActionFeatureExpander
  A_indices::Dict{T,Int}
  #=
  function NullFeatureExpander{T}(A::DiscreteActionSpace{T})
    self = new()
    self.A_indices = Dict{T,Int}([a=>i for (i,a) in enumerate(domain(A))])
    return self
  end
  =#
end
NullFeatureExpander{T}(A::DiscreteActionSpace{T}) = NullFeatureExpander(Dict{T,Int}([a=>i for (i,a) in enumerate(domain(A))]))

#NOTE: this one assumes binary sparse features
function expand{T}(exp::ActionFeatureExpander,phi::Union{SparseMatrixCSC{Float64,Int},SparseMatrixCSC{Int,Int}},a::T)
  active_indices = find(phi)
  nb_feat = length(phi)
  active_indices += nb_feat*(exp.A_indices[a]-1)
  return sparsevec(active_indices,ones(length(active_indices)),nb_feat*length(exp.A_indices))
end

#NOTE: this one assumes dense features which might not be binary
function expand{T}(exp::ActionFeatureExpander,phi::Union{Array{Float64,1},Array{Int,1}},a::T)
  _phi = zeros(length(phi)*length(exp.A_indices))
  _phi[1+length(phi)*(exp.A_indices[a]-1):length(phi)*exp.A_indices[a]] = phi
  return _phi
end
expand2{T}(exp::FeatureExpander,phi::RealVector,a::T) = expand(exp,expand(exp,phi),a)

update!(::NullFeatureExpander,phi::RealVector,del::Float64) = 0

type iFDDExpander{T} <: ActionFeatureExpander
  A_indices::Dict{T,Int}
  learned_features::Array{Tuple{Int,Int},1}
  learned_feature_set::Set{Tuple{Int,Int}}
  err_dict::Dict{Tuple{Int,Int},Float64}
  app_dict::Dict{Tuple{Int,Int},Int}
  xi::Float64 #[0.1,0.2,0.5] cutoff criterion
  #other stuff

end

function iFDDExpander{T}(A::DiscreteActionSpace{T};xi::Float64=0.5)
  #self = new()
  A_indices = Dict{T,Int}([a=>i for (i,a) in enumerate(domain(A))])
  learned_features = Tuple{Int,Int}[]
  learned_feature_set = Set{Tuple{Int,Int}}()
  err_dict = Dict{Tuple{Int,Int},Float64}()
  app_dict = Dict{Tuple{Int,Int},Int}()
  return iFDDExpander(A_indices,learned_features,learned_feature_set,err_dict,app_dict,xi)
end

function expand(expander::iFDDExpander,phi::RealVector)
  _phi = spzeros(length(expander.learned_features),1)
  active_indices = Set(find(phi))
  offset = length(phi)
  phi = vcat(phi,_phi)
  #NOTE: order matters for learned features!
  for (k,(i,j)) in enumerate(expander.learned_features)
    if (i in active_indices) && (j in active_indices)
      #(phi_copy[i] > 0) && (phi_copy[j] > 0)
      #NOTE: i think the issue is here--how to do sorted power set?
      phi[offset+k] = 1.
      push!(active_indices,offset+k)
      phi[i] = 0
      phi[j] = 0
      #i think tihs works?
    end
  end
  return phi
end

#TODO: figure out a way to make sure the state representation doesnt explode
#NOTE: what needs to happen is i deactivate features that are subfeatures to an existing feature
function update!(expander::iFDDExpander,phi::RealVector,del::Float64)
  active_indices = find(phi)
  nb_new_feat = 0
  for i in active_indices
    for j in active_indices
      #enforce ordering on (i,j) pairs
      #TODO: maintain set of learned pairs for collision detection
      if (i >= j) || ((i,j,) in expander.learned_feature_set)
        continue
      end
      err = get(expander.err_dict,(i,j),0.) + del
      expander.err_dict[(i,j)] = err
      count = get(expander.app_dict,(i,j),0) + 1
      expander.app_dict[(i,j)] = count
      if abs(err)/sqrt(count) > expander.xi
        push!(expander.learned_features,(i,j))
        push!(expander.learned_feature_set,(i,j))
        #delete from err_dict, app_dict?
        delete!(expander.err_dict,(i,j))
        delete!(expander.app_dict,(i,j))
        nb_new_feat += 1
      end
    end #j
  end #i
  return nb_new_feat
end

#pad!(x::SparseMatrixCSC,nb_new_feat::Int) = x.m += nb_new_feat
#pad!{T}(x::Array{T,1},nb_new_feat::Int) = append!(x,zeros(T,nb_new_feat))

function pad!(x::SparseMatrixCSC,nb_new_feat::Int,interval_length::Int)
  assert(mod(length(x),interval_length)== 0)
  intervals = length(x)/interval_length
  x.m += nb_new_feat*intervals
  #go through each element, greater than interval, incrememnt accordingly
  #can probably do in one pass
  for (i,ind) in enumerate(x.rowval)
    x.rowval[i] += floor(Integer,ind/interval_length)*nb_new_feat
  end
end

function pad!{T}(x::Array{T,1},nb_new_feat::Int,interval_length::Int)
  assert(mod(length(x),interval_length)== 0)
  intervals = length(x)/interval_length
  #star from the bottom
  for i = intervals:-1:1
    for j = 1:nb_new_feat
      insert!(x,convert(Int,i*interval_length+1),0.)
    end
  end
end
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
