%% This function predicts the model classification of an anomalous diffusion trajectory using CONDOR

function [ModelGuess] = ModelPrediction(MomentaInputs, dimension)

    cd(['Networks_' num2str(dimension) 'D'])

    % Classification (step 1)
    x = MomentaInputs;
    x(isnan(x)) = 0; % To avoid numerical classification problems
    load(['NetworkClass' num2str(dimension) 'D_step1.mat'])
    y = net(x);

    % Classification (step 2)
    y1 = zeros(size(y));
    y1(y == max(y)) = 1;
    y1ind = vec2ind(y1);
    x1 = MomentaInputs(:,y1ind==1 | y1ind==3 | y1ind==5); % Trajectories belonging to attm, fbm and sbm
    x1(isnan(x1)) = 0; % To avoid numerical classification problems
    load(['NetworkClass' num2str(dimension) 'D_step2.mat'])
    y2 = net(x1);
    y(1:2:5,y1ind==1 | y1ind==3 | y1ind==5) = y2;
    y(2:2:4,y1ind==1 | y1ind==3 | y1ind==5) = 0;

    % Classification (step 3)
    y1 = zeros(size(y));
    y1(y == max(y)) = 1;
    y1ind = vec2ind(y1);
    x1 = MomentaInputs(:,y1ind==1 | y1ind==2); % Trajectories belonging to attm and ctrw
    x1(isnan(x1)) = 0; % To avoid numerical classification problems
    load(['NetworkClass' num2str(dimension) 'D_step3.mat'])
    y2 = net(x1);
    y(1:2,y1ind==1 | y1ind==2) = y2;
    y(3:5,y1ind==1 | y1ind==2) = 0;

    ModelGuess = vec2ind(y); % Array with model prediction

    cd ..

end