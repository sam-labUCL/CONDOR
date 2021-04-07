%% Function to run the training of CONDOR's neural networks for the inference (submethod of subnetworks, step 1, Category C)

function TrainSubNetworkC(MomentaInputs, Alpha, dimension, ModelGuess, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)

    % Filename for the trained network
    filename_save = ['SubNetworkInfC' num2str(dimension) 'D']; 
    
    % Define inputs and targets to train the network

    MomentaInputs(end+1,:) = ModelGuess; % ModelGuess added as input

    n_targets = 5; % Number of targets
    idxC = find((Alpha > 1.00) & (Alpha <= 1.50)); % Find trajectories belonging to category C
    AlphaC = Alpha(:,idxC); % Reference Alpha values belonging to category C

    MomentaTargets = zeros(n_targets,length(AlphaC));

    % For loop to divide the alpha values in 5 different categories: each category is a target for the network
    for nn = 1:length(AlphaC)

        if (AlphaC(nn) >= 1.05 && AlphaC(nn) <= 1.10)
            MomentaTargets(1,nn) = 1;
        end

        if (AlphaC(nn) >= 1.15 && AlphaC(nn) <= 1.20)
            MomentaTargets(2,nn) = 1;
        end

        if (AlphaC(nn) >= 1.25 && AlphaC(nn) <= 1.30)
            MomentaTargets(3,nn) = 1;
        end

        if (AlphaC(nn) >= 1.35 && AlphaC(nn) <= 1.40)
            MomentaTargets(4,nn) = 1;
        end

        if (AlphaC(nn) >= 1.45 && AlphaC(nn) <= 1.50)
            MomentaTargets(5,nn) = 1;
        end

    end

    x = MomentaInputs(:,idxC);
    x(isnan(x)) = 0; % To avoid numerical classification problems
    t = MomentaTargets;

    % Create a Pattern Recognition Network
    netAlphasub = patternnet([hiddenLayerSize hiddenLayerSize], trainFcn);

    % Setup Division of Data for Training, Validation, Testing
    netAlphasub.divideParam.trainRatio = trainDataRatio;
    netAlphasub.divideParam.valRatio =  valDataRatio;
    netAlphasub.divideParam.testRatio = testDataRatio;

    netAlphasub.trainParam.max_fail = 100;
    netAlphasub.trainParam.epochs = 10000;

    % Train the Network
    [netAlphasub,tr] = train(netAlphasub,x,t);

    % Test the Network
    y = netAlphasub(x);
    e = gsubtract(t,y);
    performance = perform(netAlphasub,t,y);
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);

    % Confusion matrix plot
    figure()
    plotconfusion(t,y)

    % Save the network
    cd(['Networks_' num2str(dimension) 'D'])
    save(filename_save, 'netAlphasub', 'performance')
    cd ..

end