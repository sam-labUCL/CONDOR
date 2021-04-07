%% Function to run the training of CONDOR's neural networks for the inference (submethod of subnetworks, step 1, Category A)

function TrainSubNetworkA(MomentaInputs, Alpha, dimension, ModelGuess, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)

    % Filename for the trained network
    filename_save = ['SubNetworkInfA' num2str(dimension) 'D'];
 
    % Define inputs and targets to train the network

    MomentaInputs(end+1,:) = ModelGuess; % Add ModelGuess as input to the inputs used for classification

    n_targets = 5; % Number of targets
    idxA = find((Alpha >= 0.05) & (Alpha <= 0.50)); % Find trajectories belonging to category A
    AlphaA = Alpha(:,idxA); % Reference Alpha values belonging to category A

    MomentaTargets = zeros(n_targets,length(AlphaA));

    % For loop to divide the alpha values in 5 different categories: each category is a target for the network
    for nn = 1:length(AlphaA)

        if (AlphaA(nn) >= 0.05 && AlphaA(nn) <= 0.10)
            MomentaTargets(1,nn) = 1;
        end

        if (AlphaA(nn) >= 0.15 && AlphaA(nn) <= 0.20)
            MomentaTargets(2,nn) = 1;
        end

        if (AlphaA(nn) >= 0.25 && AlphaA(nn) <= 0.30)
            MomentaTargets(3,nn) = 1;
        end

        if (AlphaA(nn) >= 0.35 && AlphaA(nn) <= 0.40)
            MomentaTargets(4,nn) = 1;
        end

        if (AlphaA(nn) >= 0.45 && AlphaA(nn) <= 0.50)
            MomentaTargets(5,nn) = 1;
        end

    end

    x = MomentaInputs(:,idxA);
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