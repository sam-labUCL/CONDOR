%% Function to run the training of CONDOR's neural networks for the inference (submethod of subnetworks, step 1, Category D)

function TrainSubNetworkD(MomentaInputs, Alpha, dimension, ModelGuess, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)

    % Filename for the trained network
    filename_save = ['SubNetworkInfD' num2str(dimension) 'D'];
    
    % Define inputs and targets to train the network

    MomentaInputs(end+1,:) = ModelGuess; % ModelGuess added as input

    n_targets = 5; % Number of targets
    idxD = find((Alpha > 1.50) & (Alpha <= 2.00)); % Find trajectories belonging to category D
    AlphaD = Alpha(:,idxD); % Reference Alpha values belonging to category D

    MomentaTargets = zeros(n_targets,length(AlphaD));

    % For loop to divide the alpha values in 5 different categories: each category is a target for the network
    for nn = 1:length(AlphaD)

        if (AlphaD(nn) >= 1.55 && AlphaD(nn) <= 1.60)
            MomentaTargets(1,nn) = 1;
        end

        if (AlphaD(nn) >= 1.65 && AlphaD(nn) <= 1.70)
            MomentaTargets(2,nn) = 1;
        end

        if (AlphaD(nn) >= 1.75 && AlphaD(nn) <= 1.80)
            MomentaTargets(3,nn) = 1;
        end

        if (AlphaD(nn) >= 1.85 && AlphaD(nn) <= 1.90)
            MomentaTargets(4,nn) = 1;
        end

        if (AlphaD(nn) >= 1.95 && AlphaD(nn) <= 2.00)
            MomentaTargets(5,nn) = 1;
        end

    end

    x = MomentaInputs(:,idxD);
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