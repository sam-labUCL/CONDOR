%% Function to run the training of CONDOR's neural networks for the inference (submethod of models, step 1)

function TrainNetwork_step1(MomentaInputs, Alpha, dimension, ModelGuess, m, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)

    % Select the correct classification model: values between 1 and 5
    model = m; 

    % Filename for the trained network
    filename_save = ['NetworkInf' num2str(dimension) 'D_Model' num2str(model)]; 
    
    % Define the number of alpha categories to discern based on the model m 
    if model == 1 || model == 2 || model == 4
        cat_alpha = 5;
    else
        cat_alpha = 2;
    end

    % Define inputs and targets to train the network (based on model m)

    MomentaInputs(end+1,:) = ModelGuess; % Add ModelGuess as input to the inputs used for classification
    idx1 = find(MomentaInputs(end,:) == model); % Trajectories belonging to model 

    n_traj = length(idx1);

    Alpha_guess = Alpha(idx1);

    x = MomentaInputs(1:end-1,idx1); % ModelGuess is removed from the inputs
    x(isnan(x)) = 0; % To avoid numerical classification problems
    MomentaTargets = zeros(cat_alpha,n_traj); 

    for nn = 1:n_traj

        % Define targets based on alpha values allowed by model 1 and 2
        if model == 1 || model == 2 

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

        % Define targets based on alpha values allowed by model 3 and 5 using large alpha range
        if model == 3 || model == 5

            if (Alpha_guess(nn) >= 0.05 && Alpha_guess(nn) <= 1) 
                 MomentaTargets(1,nn) = 1;
            end

            if (Alpha_guess(nn) >= 1.05 && Alpha_guess(nn) <= 2) 
                MomentaTargets(2,nn) = 1;
            end

        end

        % Define targets based on alpha values allowed by model 4 
        if model == 4

            if (Alpha_guess(nn) <= 1.15) 
                 MomentaTargets(1,nn) = 1;
            end

            if (Alpha_guess(nn) >= 1.2 && Alpha_guess(nn) <= 1.35)
                 MomentaTargets(2,nn) = 1;
            end

            if (Alpha_guess(nn) >= 1.4 && Alpha_guess(nn) <= 1.55)
                 MomentaTargets(3,nn) = 1;
            end

            if (Alpha_guess(nn) >= 1.6 && Alpha_guess(nn) <= 1.75)
                 MomentaTargets(4,nn) = 1;
            end

            if (Alpha_guess(nn) >= 1.8) 
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