%% Function to run the training of CONDOR's neural networks for classification (step 2)

function TrainNetwork_step2(MomentaInputs, Model, dimension, trainFcn, hiddenLayerSize, trainDataRatio, valDataRatio, testDataRatio)
    
    % Filename for the trained network
    filename_save = ['NetworkClass' num2str(dimension) 'D_step2']; 

    % Prepare inputs
    x = MomentaInputs(:,Model==1 | Model==3 | Model==5); % Select inputs of attm (1), fbm (3) and sbm (5) trajectories only
    x(isnan(x)) = 0; % To avoid numerical classification problems
    t = Model(Model==1 | Model==3 | Model==5); % Targets for the network
    t(t==3) = 2;
    t(t==5) = 3;
    t = ind2vec(t);

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
    performance = perform(net,t,y);
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