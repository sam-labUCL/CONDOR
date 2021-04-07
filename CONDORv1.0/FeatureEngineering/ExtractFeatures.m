%% This function extracts the statistical features that will become the inputs of
% CONDOR neural networks for training and testing.

function [MomentaInputs] = ExtractFeatures(traj, dimension, size)

    n_traj = length(traj);                      % Number of trajectories

    n_inputs = dimension*92 + 1;                % Number of features to extract
    MomentaInputs = zeros(n_inputs, n_traj);    % Initialization of the array with the inputs for the networks

    N = 0;                                      % Counter of analized trajectories
    for nn = 1:n_traj

        % Calculate inputs
        MomentaInputs(:,nn) = calc_inputs(traj{nn},dimension,n_inputs);

        if mod(nn,1000) == 1
            disp([num2str(100*N/size) '% complete: ' num2str(N) ' extracted trajectories out of ' num2str(size)])
            N = N + 1000;
        end

    end

    disp([num2str(100) '% complete: ' num2str(N) ' extracted trajectories out of ' num2str(size)])

end