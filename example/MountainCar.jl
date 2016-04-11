#MountainCar.jl
# holds things in a file

push!(LOAD_PATH,joinpath("..","src"))
using ReinforcementLearning

#Base Model
type MtnCarModel <: Model
    cost::Float64
end
MtnCarModel() = MtnCarModel(-1.)


#init
init2(m::MtnCarModel,rng::AbstractRNG) = [1.8*rand()-1.2;0.14*rand()-0.07]
init1(m::MtnCarModel,rng::AbstractRNG) = [-0.5;0.]

#next
function next_state(rng::AbstractRNG,model::MtnCarModel,s::Array{Float64,1},a::Float64)
    x,v = s
    v_ = v + a*0.001+cos(3*x)*-0.0025
    v_ = max(min(0.07,v_),-0.07)
    x_ = x+v_
    #inelastic boundary
    if x_ < -1.2
        x_ = -1.2
        v_ = 0.
    end
    return [x_;v_]
end

#isterminal
isterminal(rng::AbstractRNG,m::MtnCarModel,s::Array{Float64,1},a::Float64) = s[1] >= 0.5

#Reward
reward(rng::AbstractRNG,m::MtnCarModel,s::Array{Float64,1},a::Float64) = isterminal(rng,m,s,a) ? 0.: m.cost

bbm = BlackBoxModel(MtnCarModel(),init1,next_state,reward,isterminal)

nb_bins = 20
function disjoint_feature_function(s::Array{Float64,1})
    x_ind = ReinforcementLearning.bin(s[1],-1.2,0.6,nb_bins)
    v_ind = ReinforcementLearning.bin(s[2],-0.07,0.07,nb_bins)
    ind = [x_ind; v_ind + nb_bins]
    #ind = [nb_bins*(x_ind - 1) + v_ind + 1]
    return sparsevec(ind,ones(length(ind)),nb_bins+nb_bins)
    #return sparsevec(ind,ones(length(ind)),nb_bins*nb_bins)
end

function joint_feature_function(s::Array{Float64,1})
    x_ind = ReinforcementLearning.bin(s[1],-1.2,0.6,nb_bins)
    v_ind = ReinforcementLearning.bin(s[2],-0.07,0.07,nb_bins)
    #ind = [x_ind; v_ind + nb_bins]
    ind = [nb_bins*(x_ind - 1) + v_ind + 1]
    #return sparsevec(ind,ones(length(ind)),nb_bins+nb_bins)
    return sparsevec(ind,ones(length(ind)),nb_bins*nb_bins)
end

import Base.convert
import Base.vec

vec(x::Float64) = Float64[x]

Float64(x::Array{Float64,1}) = x[1]
