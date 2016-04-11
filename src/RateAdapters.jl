type NullRateAdapter <: RateAdapter
  lr0::Float64
end
learning_rate!(ra::NullRateAdapter,t::Int,ep::Int,phi::RealVector,updater::UpdaterParam) = ra.lr0 #TODO figure out the rest of the signature

type ExponentialRateAdapter <: RateAdapter
  lr0::Float64
  k::Union{Float64,Int}
  t::Int
  function ExponentialRateAdapter(;lr::Float64=0.1,
                                    k::Int=1000)
    self = new()
    self.lr0 = lr
    self.k = k
    self.t = 0
    return self
  end
end
function learning_rate!(ra::ExponentialRateAdapter,t::Int,ep::Int,phi::RealVector,updater::UpdaterParam)
  ra.t += 1
  return ra.lr0*exp(-ra.k*ra.t)
end

type StepRateAdapter <: RateAdapter
  lr0::Float64
  t::Int
  t0::Int #halving interval
  function StepRateAdapter(;lr::Float64=0.1,
                                    t0::Int=1000)
    self = new()
    self.lr0 = lr
    self.t0 = t0
    self.t = 0
    return self
  end
end
function learning_rate!(ra::StepRateAdapter,t::Int,ep::Int,phi::RealVector,updater::UpdaterParam)
  ra.t+= 1
  k = floor(ra.t/ra.t0)
  return ra.lr0/(2^k)
end

type InverseRateAdapter <: RateAdapter
  lr0::Float64
  k::Float64
  t::Int
  function InverseRateAdapter(;lr::Float64=0.1,
                                    k::Int=1000)
    self = new()
    self.lr0 = lr
    self.k = k
    self.t = 0
    return self
  end
end
function learning_rate!(ra::InverseRateAdapter,t::Int,ep::Int,phi::RealVector,updater::UpdaterParam)
  ra.t += 1
  return ra.lr0/(1+ra.k*ra.k)
end

type iFDDRateAdapter <: RateAdapter
  lr0::Float64 #[0.1,1]
  N0::Float64 #[100,1000,1e6]
  function iFDDRateAdapter(;lr::Float64=0.1,
                                    N0::Int=1000)
    self = new()
    self.lr0 = lr
    self.N0 = N0
    return self
  end
end
function learning_rate!(ra::iFDDRateAdapter,t::Int,ep::Int,phi::RealVector,updater::UpdaterParam)
  k = countnz(phi)
  return ra.lr0*(ra.N0+1)/(k*(ra.N0+ep^1.1))
end
