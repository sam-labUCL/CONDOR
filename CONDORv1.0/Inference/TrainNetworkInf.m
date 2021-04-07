%% Function to run the training of CONDOR's neural networks for inference

function TrainNetworkInf(MomentaInputs, Alpha, dimension, ModelGuess, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
 
    % Train network for Models_method

    cd('Models_method')

    % Network training for all the models
    for m = 1:5
        TrainNetwork_step1(MomentaInputs, Alpha, dimension, ModelGuess, m, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
        pause(5)
        close all;
    end

    % Additional network training for fbm and sbm
    mdls = [3 5];

    for m = 1:length(mdls)
        for k = 1:2
            TrainNetwork_step2(MomentaInputs, Alpha, dimension, ModelGuess, mdls(m), k, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
            pause(5)
            close all;
        end
    end

    cd ..

    % Train network for Subnetworks_method 

    cd('Subnetworks_method')

    % Call to function to train CONDOR's neural networks for inference with subnetworks_method (step 1)
    TrainMainNetwork(MomentaInputs, Alpha, dimension, ModelGuess, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    pause(5)
    close all;
    % Call to function to train CONDOR's neural networks for inference with submethod of subnetworks_method (step 1, category A)
    TrainSubNetworkA(MomentaInputs, Alpha, dimension, ModelGuess, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    pause(5)
    close all;
    % Call to function to train CONDOR's neural networks for inference with submethod of subnetworks_method (step 1, category B)
    TrainSubNetworkB(MomentaInputs, Alpha, dimension, ModelGuess, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    pause(5)
    close all;
    % Call to function to train CONDOR's neural networks for inference with submethod of subnetworks_method (step 1, category C)
    TrainSubNetworkC(MomentaInputs, Alpha, dimension, ModelGuess, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    pause(5)
    close all;
    % Call to function to train CONDOR's neural networks for inference with submethod of subnetworks_method (step 1, category D)
    TrainSubNetworkD(MomentaInputs, Alpha, dimension, ModelGuess, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    pause(5)
    close all;

    cd ..

end
