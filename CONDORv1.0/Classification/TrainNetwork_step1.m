%% Function to run the training of CONDOR's neural networks for classification (step 1)

function TrainNetwork_step1(MomentaInputs, Model, dimension, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    
    % Filename for the trained network
    filename_save = ['NetworkClass' num2str(dimension) 'D_step1'];

    %  Define targets 
    n_category = 5; % Number of the models for the classification of trajectories (attm, ctrw, fbm, lw, sbm)
    n_traj = length(MomentaInputs);
    MomentaTargets = zeros(n_category, n_traj);

    for nn = 1:n_traj
        MomentaTargets(Model(nn),nn) = 1;
    end

    % Prepare inputs
    x = MomentaInputs; % Inputs for the network
    x(isnan(x)) = 0; % To avoid numerical classification problems
    t = MomentaTargets; % Targets for the network

    % Create a Pattern Recognition Network
    net = patternnet([hiddenLayerSize hiddenLayerSize], trainFcn);

    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = trainDataRatio;
    net.divideParam.valRatio =  valDataRatio;
    net.divideParam.testRatio = testDataRatio;

    if strcmp(trainFcn, 'trainscg') == 1
        net.trainParam.max_fail = 100;
        net.trainParam.epochs = 10000;
    end

    % Train the Network
    [net,tr] = train(net,x,t);

    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    performance = perform(net,t,y)
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);

    % Plots
    figure, plotconfusion(t,y)

    % Save network
    cd(['Networks_' num2str(dimension) 'D'])
    save(filename_save, 'net', 'performance')
    cd ..

end