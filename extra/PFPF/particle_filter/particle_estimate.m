function [estimate,ml_weights] = particle_estimate(log_weights,particles,maxilikeSAP,maxilikemode,approxexpflag)
% Form estimate based on weighted set of particles
%
% log_weights: logarithmic weights [Nx1]
% particles: the state values of the particles [dim x N] (state dimension x
% number of particles)
% maxilikeSAP: use a reduced set of particles specified by this number
% maxilikemode: 'm'= median, 'a' = weighted means
% approxexpflag: use a polynomial approximation to the exp function
%
log_weights = log_weights-max(log_weights);

ml_weights = exp(log_weights(:));
ml_weights = ml_weights/sum(ml_weights);
estimate = particles*ml_weights;

end