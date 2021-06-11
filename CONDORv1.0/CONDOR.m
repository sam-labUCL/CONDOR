%% Load cell array traj with trajectories to analyze and references if provided
[filename,filepath] = uigetfile('*.mat');
load(filename)

prompt = 'What is the dimension (1,2,3)? ';         % Ask for dimension of the trajectory (1, 2, 3)
d = input(prompt);    % 1 for traj_example

%% Ask user what to do (networks training and/or prediction)
prompt = 'Would you like to train CONDOR networks for classification (y/n)? ';
choice1 = input(prompt,'s');

prompt = 'Would you like to predict the model classification with CONDOR (y/n)? ';
choice2 = input(prompt,'s');

prompt = 'Would you like to train CONDOR networks for inference (y/n)? ';
choice3 = input(prompt,'s');

prompt = 'Would you like to predict the anomalous diffusion coefficient with CONDOR (y/n)? ';
choice4 = input(prompt,'s');

%% Creation of struct Dataset that contains all the information of the trajectories to analyze
if exist('Model','var') == 1
    Dataset.model_ref = 'Model';                    % Array of model references if provided
else
    Dataset.model_ref = 'none';
end

if exist('Alpha','var') == 1
    Dataset.alpha_ref = 'Alpha';                    % Array of alpha references if provided
else
    Dataset.alpha_ref = 'none';
end

Dataset.dimension = d;                             % Data dimension (1, 2 or 3)
Dataset.size = length(traj);                       % Number of trajectories

%% Feature engineering: calculate network inputs
if exist('MomentaInputs', 'var') == 0
    disp('Statistical analysis starting...')
    cd FeatureEngineering
    MomentaInputs = ExtractFeatures(traj, Dataset.dimension, Dataset.size);
    cd ..
    
    varTosave = {'filename', 'traj', 'Dataset', 'MomentaInputs', 'Alpha', 'Model'};
    
    for i = 1:length(varTosave)
        if exist(varTosave{i}, 'var')
            save(filename, varTosave{i}, '-append')
        end
    end
end
    
%% Call to function to train the networks for classification (reference needed)
if choice1 == 'y'
    disp('Training for classification starting...')
    cd Classification
    
    % Define parameters for training 
    trainFcn = 'trainscg';      % Training function name
    hiddenLayerSize = 20;       % Size of the hidden layers
    trainDataRatio = 70/100;    % Division of data for training
    valDataRatio =  15/100;     % Division of data for validation
    testDataRatio = 15/100;     % Division of data for test
    
    TrainNetworkClass(MomentaInputs, Model, Dataset.dimension, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    cd ..
end

%% Call to function to predict the model (ModelGuess) of the trajectories (1 = attm, 2 = ctrw, 3 = fbm, 4 = lw, 5 = sbm)
if ((choice2 == 'y') || (choice3 == 'y') || (choice4 == 'y'))
    cd Classification
    ModelGuess = ModelPrediction(MomentaInputs, Dataset.dimension);
    cd ..
end

%% Call to function to train the networks for inference (reference and ModelGuess needed)
if choice3 == 'y'
    disp('Training for inference starting...')
    cd Inference
    
    % Define parameters for training 
    trainFcn = 'trainscg';      % Training function name (scaled conjugate gradient backpropagation)
    hiddenLayerSize = 20;       % Size of the hidden layers
    trainDataRatio = 70/100;    % Division of data for training
    valDataRatio =  15/100;     % Division of data for validation
    testDataRatio = 15/100;     % Division of data for test
    
    TrainNetworkInf(MomentaInputs, Alpha, Dataset.dimension, ModelGuess, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    cd ..
end

%% Call to function to predict (ModelGuess needed) the anomalous diffusion coefficient (AlphaGuess)
if choice4 == 'y'
    cd Inference
    AlphaGuess = AlphaPrediction(MomentaInputs, Dataset.dimension, Dataset.size, ModelGuess);
    cd ..
end

disp('Process completed.')