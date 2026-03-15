function tracking_output = run_one_trial(ps)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the interface used to run different filtering algorithms.
%
% Inputs:
% ps: structure with filter and simulation parameters
%
% Output:
% tracking_output: a struct that contains outputs from different filters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tracking_output = [];

trial_ix = ps.trial_ix;

ps.x = ps.x_all{ceil(trial_ix/ps.setup.nAlg_per_track)};
y = ps.y_all{ceil(trial_ix/ps.setup.nAlg_per_track)};

if ismember('PFPF_LEDH',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    ps_new = ps;
    ps_new.setup.pf_type = 'LEDH';
    tracking_output.PFPF_LEDH = PFPF(ps_new,y);
end

if ismember('PFPF_EDH',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    ps_new = ps;
    ps_new.setup.pf_type = 'EDH';
    tracking_output.PFPF_EDH = PFPF(ps_new,y);
end

if ismember('LEDH',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    ps_new = ps;
    ps_new.setup.pf_type = 'LEDH';
    ps_new.setup.PFPF = false;
    tracking_output.LEDH = DH_ExactFlow_Filter(ps_new,y);
end

if ismember('EDH',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    ps_new = ps;
    ps_new.setup.pf_type = 'EDH';
    ps_new.setup.PFPF = false;
    tracking_output.EDH = DH_ExactFlow_Filter(ps_new,y);
end

if ismember('GPFIS',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    tracking_output.GPFIS = GaussianFlowParticleFilter(ps,y);
end

if ismember('SmHMC',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    tracking_output.SmHMC = SmHMC_interface(ps,y);
end

if ismember('BPF',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    tracking_output.BPF = BootstrapParticleFilter(ps,y);
end

if ismember('EKF',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    ps_new = ps;
    ps_new.setup.kflag = 'EKF1';
    tracking_output.EKF = EKF_Filter(ps_new,y);
end

%% Changing the particle number for a few algorithms, as evaluated in [li2016]
if ismember('PFPF_EDH_1e4',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    ps_new = ps;
    ps_new.setup.nParticle = 1e4;
    ps_new.setup.pf_type = 'EDH';
    tracking_output.PFPF_EDH_1e4 = PFPF(ps_new,y);
end

if ismember('EDH_1e4',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    ps_new = ps;
    ps_new.setup.nParticle = 1e4;
    ps_new.setup.pf_type = 'EDH';
    ps_new.setup.PFPF = false;
    tracking_output.EDH_1e4 = DH_ExactFlow_Filter(ps_new,y);
end

if ismember('BPF_1e5',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    ps_new = ps;
    ps_new.setup.nParticle = 1e5;
    tracking_output.BPF_1e5 = BootstrapParticleFilter(ps_new,y);
end

if ismember('BPF_1e6',ps.setup.algs_executed)
    rng(ps.setup.random_seeds(trial_ix),'twister'); 
    ps_new = ps;
    ps_new.setup.nParticle = 1e6;
    tracking_output.BPF_1e6 = BootstrapParticleFilter(ps_new,y);
end


%% Displaying estimation errors for all algorithms
disp('------------------------------------------------------------')
disp('------------------------------------------------------------')
disp(['Displaying results for all algorithms, at Trial ', num2str(trial_ix), ':']);
alg_field_names = fieldnames(tracking_output);
for field_ix = 1:length(alg_field_names)
    field_name = alg_field_names{field_ix};
    calculateErrors(tracking_output.(field_name),ps,field_name);
end
disp('------------------------------------------------------------')
disp('------------------------------------------------------------')
end