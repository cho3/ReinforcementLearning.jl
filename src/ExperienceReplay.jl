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

#TODO prioritized experience replay
