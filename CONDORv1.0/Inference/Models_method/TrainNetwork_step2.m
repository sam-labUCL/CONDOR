%% Function to run the training of CONDOR's neural networks for the inference (submethod of models, step 2)

function TrainNetwork_step2(MomentaInputs, Alpha, dimension, ModelGuess, m, k, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)

    % Train networks for alpha exponent inference in trajectories corresponding to the classification model m
    model = m; % Either 3 or 5
    netw = k;  % Either 1 or 2 

    % Filename for the trained network
    filename_save = ['NetworkInf' num2str(dimension) 'D_Model' num2str(model) '_' num2str(netw)]; 

    % Define inputs and targets to train the network (based on model m)
   
    MomentaInputs(end+1,:) = ModelGuess; % Add ModelGuess as input to the inputs used for classification
    idx1 = find(MomentaInputs(end,:) == model); % Trajectories belonging to model 

    % Select range of alpha for model 3 and model 5
    if netw == 1
        idx2 = find(Alpha(idx1) <= 1);
    end

    if netw == 2
        idx2 = find(Alpha(idx1) > 1);
    end

    Alpha_guess = Alpha(idx1(idx2));
    n_traj = length(idx2);

    x = MomentaInputs(1:end-1,idx1(idx2)); % ModelGuess is removed from the inputs
    x(isnan(x)) = 0; % To avoid numerical classification problems

    MomentaTargets = zeros(5,n_traj); 

    for nn = 1:n_traj

        % Targets for alpha values <= 1
        if netw == 1

            if (Alpha_guess(nn) >= 0.05 && Alpha_guess(nn) <= 0.2) 
                 MomentaTargets(1,nn) = 1;
            end

            if (Alpha_guess(nn) >= 0.25 && Alpha_guess(nn) <= 0.4)
                 MomentaTargets(2,nn) = 1;
            end

            if (Alpha_guess(nn) >= 0.45 && Alpha_guess(nn) <= 0.6) 
                 MomentaTargets(3,nn) = 1;
            end

            if (Alpha_guess(nn) >= 0.65 && Alpha_guess(nn) <= 0.8) 
                 MomentaTargets(4,nn) = 1;
            end

            if (Alpha_guess(nn) >= 0.85)
                MomentaTargets(5,nn) = 1;
            end   

        end

        % Targets for alpha values > 1
        if netw == 2

            if (Alpha_guess(nn) <= 1.2)
                 MomentaTargets(1,nn) = 1;
            end

            if (Alpha_guess(nn) >= 1.25 && Alpha_guess(nn) <= 1.4)
                 MomentaTargets(2,nn) = 1;
            end

            if (Alpha_guess(nn) >= 1.45 && Alpha_guess(nn) <= 1.6) 
                 MomentaTargets(3,nn) = 1;
            end

            if (Alpha_guess(nn) >= 1.65 && Alpha_guess(nn) <= 1.8)
                 MomentaTargets(4,nn) = 1;
            end

            if (Alpha_guess(nn) >= 1.85 && Alpha_guess(nn) <=2) 
                MomentaTargets(5,nn) = 1;
            end   

        end

    end

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
    figure, plotconfusion(t,y)

    % Save the network
    cd(['Networks_' num2str(dimension) 'D'])
    save(filename_save, 'netAlpha', 'performance')
    cd ..

end