#Annealers.jl
#should probably switch this name to rate adapters

#Vanilla annealer
type NullAnnealer <: AnnealerParam end
AnnealerParam() = NullAnnealer()
anneal!(::AnnealerParam,dw::Union{RealVector,RealMatrix},lr::Float64,idx::Int) = lr*dw


#TODO: adapt everything to real matrices

#NOTE:"Taken from cs231n.github.io"
#TODO: make sure this doesn't break from minibatcher returning zeros()
#Momentum Update
type MomentumAnnealer <:AnnealerParam
  v::Dict{Int,RealVector} #having just  single array may be insufficient for more complex things
  mu::Dict{Int,Float64} #[0.5, 0.9, 0.95, 0.99]
end
function anneal!(an::MomentumAnnealer,dw::RealVector,lr::Float64,idx::Int)
  an.v[idx] = an.mu[idx]*an.v[idx] - lr*dw #TODO: how to handle lr btwn stuff
  return an.v[idx]
end

#Nesterov update
type NesterovAnnealer <: AnnealerParam
  v::Dict{Int,RealVector}
  mu::Dict{Int,Float64}
end
function anneal!(an::NesterovAnnealer, dw::RealVector,lr::Float64,idx::Int)
  v_prev = an.v[idx]
  v = an.mu[idx]*an.v[idx] - lr*dw #TODO: how to handle lr btwn tuff
  return -an.mu[idx]*v_prev + (1 + an.mu[idx])*v
end

#Adagrad update
type AdagradAnnealer <: AnnealerParam
  cache::Dict{Int,RealVector}
  fuzz::Dict{Int,Float64} #1e-8
end
function anneal!(an::AdagradAnnealer,dw::RealVector,lr::Float64,idx::Int)
  an.cache[idx] += dw.^2
  return lr*dw./sqrt(an.cache[idx] + an.fuzz)
end

#Adadelta update
type AdadeltaAnnealer <: AnnealerParam
  mu::Dict{Int,Float64}
  fuzz::Dict{Int,Float64}
  dw2::Dict{Int,RealVector}
  dx2::Dict{Int,RealVector}
end
function anneal!(an::AdadeltaAnnealer,dw::RealVector,lr::Float64,idx::Int)
  an.dw2[idx] = an.mu[idx]*an.dw2[idx] + (1-an.mu[idx])*(dw.^2)
  dx = dw.*sqrt(an.dx2[idx] + an.fuzz[idx])./sqrt(an.dw2[idx] + an.fuzz)
  an.dx2[idx] = an.mu[idx]*an.dx2[idx] + (1.-an.mu[idx])*(dx.^2)
  return lr*dx
end

#RMSProp update
type RMSPropAnnealer <: AnnealerParam
  cache::Dict{Int,RealVector}
  fuzz::Dict{Int,Float64} #1e-8
  decay_rate::Dict{Int,Float64} #[0.9; 0.99; 0.999]
end
function anneal!(an::RMSPropAnnealer,dw::RealVector,lr::Float64,idx::Int)
  an.cache[idx] = an.decay_rate[idx]*an.cache[idx] + (1.-an.decay_rate[idx])*(dw.^2)
  return lr*dw./sqrt(an.cache[idx] + an.fuzz[idx])
end

#ADAM update
type AdamAnnealer <: AnnealerParam
  mu::Dict{Int,Float64} #0 <= 0.9 < 1
  nu::Dict{Int,Float64} #might be bad naming convention, 0<0.999 <1
  u::Dict{Int,RealVector}
  v::Dict{Int,RealVector}
  fuzz::Dict{Int,Float64} #1e-8
  t::Int #init 0
end
function anneal!(an::AdamAnnealer,dw::RealVector,lr::Float64,idx::Int)
  an.t += 1
  an.v[idx] = an.mu[idx]*an.v[idx]+ (1.-an.mu[idx])*dw
  an.u[idx] = an.nu[idx]*an.u[idx] + (1.-an.nu[idx])*(dw.^2)
  v_ = an.v[idx]./(1-an.mu[idx]^an.t)
  u_ = an.u[idx]./(1.-an.nu[idx]^an.t)
  return lr*v_./(sqrt(u_) + an.fuzz[idx])
end
