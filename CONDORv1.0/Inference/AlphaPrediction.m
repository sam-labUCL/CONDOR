%% This function predicts the value of the alpha exponent of an anomalous diffusion trajectory using CONDOR

function [AlphaGuess] = AlphaPrediction(MomentaInputs, dimension, sizeAlpha, ModelGuess)

    % Prediction of alpha_1 with the submethod of the models
    Alpha_models = AlphaPrediction_Models(MomentaInputs, dimension, sizeAlpha, ModelGuess);

    % Prediction of alpha_2 with the submethod of the subnetworks 
    Alpha_subnets = AlphaPrediction_Subnetworks(MomentaInputs, dimension, sizeAlpha, ModelGuess);

    % Prediction of alpha as mean of Alpha guess values from models_method and subnetworks_method
    AlphaGuess = mean([Alpha_subnets; Alpha_models]);

end