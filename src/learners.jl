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
minibatch!(mb::NullMinibatcher,dw::Array{Float64,1}) = dw

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
#TODO: RMSProp
#TODO: momentum
#TODO: nesterov
#TODO: adagrad/adadelta
