#learners.jl
#this might be broken into learningrate.jl and minibatch.jl and experiencereplay.jl in the future


abstract AnnealerParam

#Vanilla annealer
type NullAnnealer <: AnnealerParam end
anneal!(::AnnealerParam,dw::Array{Float64,1}) = dw



abstract ExperienceReplayer
type NullExperienceReplayer <:ExperienceReplayer end
replay!(er::NullExperienceReplayer,phi,r,phi_) = phi,r,phi_#placeholder

#TODO: parameterize this?
#TODO: or make it just a pair of feature vectors + reward
type Experience{T}
  phi::Array{Union{Float64,Int},1}
  a::T
  r::Float64
  phi_::Array{Union{Float64,Int},1}
  a_::T
end
remember(e::Experience) = e.phi,e.a,e.r,e.phi_,e.a_
type UniformExperienceReplayer <: ExperienceReplayer
  memory::Array{Experience,1}
  nb_mem::Int
  rng::AbstractRNG
end
function replay!{T}(er::UniformExperienceReplayer,
                  phi::::Array{Union{Float64,Int},1},
                  a::T,
                  r::Float64,
                  phi_::Array{Union{Float64,Int},1},
                  a_::T)
  e = Experience(phi,a,r,phi_,a_)
  if length(er.memory) < er.nb_mem
    push!(er.memory,e)
  else
    ind = rand(er.rng,1:er.nb_mem)
    #is it more memory efficient to delete somehow first?
    er.memory[ind] = e
  end
  ind = rand(er.rng,1:er.nb_mem)
  return remember(er.mem[ind])
end


##################################
abstract Minibatcher
type NullMinibatcher <: Minibatcher end
minibatch!(mb::NullMinibatcher,dw::Array{Float64,1}) = dw

type UniformMinibatcher <: Minibatcher
  minibatch_size::Int
  dw::Array{Float64,1}
  current_minibatch_size::Int
end
function minibatch!(mb::UniformMinibatcher,dw::Array{Float64,1})
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
#TODO: RMSProp
#TODO: momentum
#TODO: nesterov
#TODO: adagrad/adadelta
