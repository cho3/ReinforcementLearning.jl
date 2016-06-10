# Define an API for regression tasks, eg. value function estimation or policies


abstract Regressor
abstract Objective # e.g. L2 loss

### Generic Interface

# Smash parameters into 1d vector
squash(::Regressor) = error("Undefined Regressor")
# Take 1D vector and change it into a form compatible with regressor type
desquash(::Regressor,x::RealVector) = error("Undefined Regressor")

# Increment regressor values with dx
#  e.g. update!(r,dx) = begin r.x += desquash(r,dx) end, assuming +(regressor) defined
update!(::Regressor, dx::RealVector) = error("Undefined Regressor")
set!(::Regressor, x::RealVector) = error("Undefined Regressor")
set!(a::Regressor, b::Regressor) = begin set!(a,squash(b)) end

# Estimate the value of whatever you're estimating
estimate(::Regressor, x::RealVector) = error("Undefined Regressor")


### Linear Regressor
# TODO split into UnivariateLinearRegressor and MultivariateLinearRegressor?

abstract LinearRegressor <: Regressor

type UnivariateLinearRegressor <: LinearRegressor
  weights::RealVector
end
UnivariateLinearRegressor() = UnivariateLinearRegressor(zeros(0)) #uninstantiated TODO

type MultivariateLinearRegressor <: LinearRegressor
  weights::RealMatrix
end
MultivariateLinearRegressor() = MultivariateLinearRegressor(zeros(0,0))

squash(lr::LinearRegressor) = vec(lr.weights)
desquash(lr::LinearRegressor, x::RealVector) = reshape(x, shape(lr.weights))

# TODO how to handle uninitialized stuff
function update!(lr::LinearRegressor, dx::RealVector)
  if length(lr.weights) = 0

  end
  lr.weights += dx
  return lr
end

function set!(lr::LinearRegressor, x::RealVector)
  lr.weights = desquash(lr, x)
  return lr
end

function estimate(r::LinearRegressor, x::RealVector)
  y = vec(r.weights'*x)
  if length(y) == 1
    y = y[1]
  end
  return y
end

# TODO need things like learning rate and crap
function train!(  o::Union{Updater,Objective},
                  r::UnivariateLinearRegressor,
                  ra::RateAdapter,
                  mb::Minibatcher,
                  gc::GradientClipper,
                  lr::Float64,
                  target::Real,
                  x::RealVector,
                  idx::Int=1) #indexes which estimator its updating
  # e.g. Q learning or L2 loss
  estimate = estimate(r,x)
  del = target - estimate
  # if L1: del = sign(del)
  dw = del * x
  dw = clip!(gc, dw, idx)

  update!(r, anneal!(ra ,minibatch!(mb,dw,idx),lr,idx))

  return r
end

function train(o::Union{Updater,Objective},r::MultivariateLinearRegressor, target::RealVector, x::RealVector)
  error("Unimplemented")
end
