%% Function to run the training of CONDOR's neural networks for classification 

function TrainNetworkClass(MomentaInputs, Model, dimension, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)

    % Call to function to train CONDOR's neural networks for classification (Step 1)
    TrainNetwork_step1(MomentaInputs, Model, dimension, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    pause(5)
    close all;
    
    % Call to function to train CONDOR's neural networks for classification (Step 2)
    TrainNetwork_step2(MomentaInputs, Model, dimension, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    pause(5)
    close all;
    
    % Call to function to train CONDOR's neural networks for classification (Step 3)
    TrainNetwork_step3(MomentaInputs, Model, dimension, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    pause(5)
    close all;

end