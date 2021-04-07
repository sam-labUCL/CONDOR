%% Function to run the training of CONDOR's neural networks for the inference (submethod of subnetworks, step 1)

function TrainMainNetwork(MomentaInputs, Alpha, dimension, ModelGuess, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)

    % Filename for the trained network
    filename_save = ['NetworkMainInf' num2str(dimension) 'D']; 

    % Define inputs and targets to train the network

    MomentaInputs(end+1,:) = ModelGuess; % Add ModelGuess as input to the inputs used for classification

    n_targets = 4; % Number of targets
    n_traj = length(MomentaInputs); % Number of trajectories

    MomentaTargets = zeros(n_targets, n_traj);

    % For loop to divide the alpha values in 4 different categories: each category is a target for the network
    for nn = 1:n_traj

        if (Alpha(nn) >= 0.05 && Alpha(nn) <= 0.50) %Category A
            MomentaTargets(1,nn) = 1;
        end

        if (Alpha(nn) > 0.50 && Alpha(nn) <= 1.00) %Category B
            MomentaTargets(2,nn) = 1;
        end

        if (Alpha(nn) > 1.00 && Alpha(nn) <= 1.50) %Category C
            MomentaTargets(3,nn) = 1;
        end

        if (Alpha(nn) > 1.50 && Alpha(nn) <= 2.00) %Category D
            MomentaTargets(4,nn) = 1;
        end

    end

    x = MomentaInputs;
    x(isnan(x)) = 0; % To avoid numerical classification problems
    t = MomentaTargets;

    % Create a Pattern Recognition Network
    netAlpha = patternnet([hiddenLayerSize hiddenLayerSize], trainFcn);

    % Setup Division of Data for Training, Validation, Testing
    netAlpha.divideParam.trainRatio = trainDataRatio;
    netAlpha.divideParam.valRatio =  valDataRatio;
    netAlpha.divideParam.testRatio = testDataRatio;

    netAlpha.trainParam.max_fail = 100;
    netAlpha.trainParam.epochs = 10000;

    % Train the Network
    [netAlpha,tr] = train(netAlpha,x,t);

    % Test the Network
    y = netAlpha(x);
    e = gsubtract(t,y);
    performance = perform(netAlpha,t,y);
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);

    % Confusion matrix plot 
    figure()
    plotconfusion(t,y)

    % Save the network
    cd(['Networks_' num2str(dimension) 'D'])
    save(filename_save, 'netAlpha', 'performance')
    cd ..
    
end