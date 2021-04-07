%% This function predicts a first value of the alpha exponent of an anomalous diffusion trajectory using CONDOR (submethod of the models)

function [Alpha_guess_models] = AlphaPrediction_Models(MomentaInputs, dimension, sizeAlpha, ModelGuess)

    MomentaInputs(end+1,:) = ModelGuess; % ModelGuess added as input initially to select the model

    Alpha_guess_models = zeros(1,length(sizeAlpha)); % Initialization of array for Alpha guess

    % Predict Alpha value according to model classification

    cd('Models_method')

    cd(['Networks_' num2str(dimension) 'D'])

    for mdl = 1:5

        idx1 = find(MomentaInputs(end,:) == mdl);  % Trajectories belongs to model mdl

        x = MomentaInputs(1:end-1,idx1); % Model guess is removed from the inputs
        x(isnan(x)) = 0;

        load(['NetworkInf' num2str(dimension) 'D_Model' num2str(mdl)])

        % Test the Network
        y_alpha = netAlpha(x);

        % Assign AlphaG as mean value of the alpha range allowed by the model
        if mdl == 1 || mdl == 2
            AlphaG = [0.2 0.4 0.6 0.8 1]-0.075;
        end

        if mdl == 3 || mdl == 5
            AlphaG = [1 2]-0.475;
        end

        if mdl == 4
            AlphaG = [1 1.2 1.4 1.6 1.8]+0.075;
        end

        % Build Alpha_guess array
        Alpha_guess = AlphaG(vec2ind(y_alpha));
        Alpha_guess_models(idx1) = Alpha_guess;

    end

    % Refine Alpha estimation for models 3 and 5

    mdl1 = [3 5];

    for m = 1:2

        for mm = 1:2

            idx1 = find(MomentaInputs(end,:) == mdl1(m));       % Trajectories belongs to model mdl1(m)
            
            % Prediction for alpha values <= 1
            if mm == 1
                idx2 = find(Alpha_guess_models(idx1) <= 1);     % Trajectories belongs to model mdl1(m) with a previous estimated alpha <= 1
                AlphaG = [0.2 0.4 0.6 0.8 1]-0.075;
            end

            % Prediction for alpha values > 1
            if mm == 2
                idx2 = find(Alpha_guess_models(idx1) > 1);      % Trajectories belongs to model mdl1(m) with a previous estimated alpha > 1
                AlphaG = [1 1.2 1.4 1.6 1.8]+0.075;
            end

            load(['NetworkInf' num2str(dimension) 'D_Model' num2str(mdl1(m)) '_' num2str(mm)])

            x = MomentaInputs(1:end-1,idx1(idx2)); % Inputs of trajectories of model mdl1(m) and with alpha <=1 or alpha > 1 (according to the previous step)
            x(isnan(x)) = 0;

            % Test the network
            y_alpha = netAlpha(x);

            % Refine Alpha_guess array
            Alpha_guess = AlphaG(vec2ind(y_alpha));
            Alpha_guess_models(idx1(idx2)) = Alpha_guess; % Correction of the Alpha_guess array build in step 1

        end

    end

    cd ..
    cd ..

end
