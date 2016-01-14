#solve.jl
#a place to hold the base solver stuff

type Solver
  lr::Float64 #initial learning relate
  nb_episodes::Int
  nb_timesteps::Int
  discount::Float64
  simRNG::AbstractRNG
  annealer::AnnealerParam #how the learning rate adapts
  updater::UpdaterParam
  mb::Minibatcher
  er::ExperienceReplayer
  function Solver(updater::UpdaterParam;
                    lr::Float64=0.01,
                    nb_episodes::Int=100,
                    nb_timesteps::Int=100,
                    discount::Float64=0.99,
                    simRNG::AbstractRNG=MersenneTwister(23894),
                    annealer::AnnealerParam=NullAnnealer(),
                    mb::Minibatcher=NullMinibatcher(),
                    er::ExperienceReplayer=NullExperienceReplayer())
    self = new()
    self.lr = lr
    self.nb_episodes = nb_episodes
    self.nb_timesteps = nb_timesteps
    self.discont = discount
    self.simRNG = simRNG
    self.annealer = annealer
    self.mb = mb
    self.er = er
    self.updater = updater
  end
end


function solve(solver::Solver,bbm::BlackBoxModel,policy::Policy)

  #maintain statistics?
  for ep = 1:solver.nb_episodes
    #episode setup stuff
    s = feature_function(init(bbm,solver.simRNG))
    a = action(policy,updater,s)
    phi = policy.feature_function(s,a)
    for t = 1:solver.nb_timesteps
      r, s_ = next(bbm,a)
      a_ = action(policy,updater,s_)
      phi_ = policy.feature_function(s_,a_)
      gamma = isterminal(bbm,a_) ? 0. : solver.discount
      #NOTE: using s,a,r,s_,a_ for maximum generality
      update!(solver.updater,solver.annealer,solver.mb,solver.er,phi,a,r,phi_,a,gamma,lr)
      if gamma == 0.
        break
      end
      #push the update frame up one time step as it were
      s = s_ #TODO: not needed?
      a = a_
      phi = phi_
    end #t

  end #ep

  #return something--policy? stats?
  return Policy(policy,solver.updater)
end
