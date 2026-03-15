function [vg,output] = initializationFilter(ps)
%% initialize filter outputs
% inputs:
% ps: structure with filter and simulation parameters
%
% Output:
% vg: a struct that contains the filter output
% output: a struct that contains the filter estimates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T = ps.setup.T; % number of time steps

output = struct('x_est',zeros(ps.setup.dimState_all,T));%,'M',zeros(ps.setup.dimState_all,T), 'PU',zeros(ps.setup.dimState_all,ps.setup.dimState_all,T));
output.x_est_unweighted = output.x_est;
% Kalman variables
vg.M = zeros(ps.setup.dimState_all,1); % EKF1/EKF2/UKF estimate
vg.PP = zeros(ps.setup.dimState_all,ps.setup.dimState_all); % EKF1/EKF2/UKF pred. covariance
vg.PU = zeros(ps.setup.dimState_all,ps.setup.dimState_all); % EKF1/EKF2/UKF upd. covariance

% ps.Initfcn: particle initialization function
[vg.xp,vg.M,vg.PU] = ps.initparams.initfcn(ps,ps.setup.nParticle); % Initialize particles, KF estimate and cov

% In the PF-PF (LEDH), we eliminate correlations between particles so that
% each particle uses potentially different PU to update the predictive
% covariance.
if ps.setup.PFPF && isfield(ps.setup,'pf_type')
    switch ps.setup.pf_type
        case 'LEDH'
            vg.PU_all = repmat(vg.PU,1,1,ps.setup.nParticle);
    end
end

vg.xp_m = vg.M;

vg.logW = zeros(1,size(vg.xp,2));

end

