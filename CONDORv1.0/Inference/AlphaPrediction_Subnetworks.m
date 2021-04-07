%% This function predicts a second value of the alpha exponent of an anomalous diffusion trajectory using CONDOR (submethod of the subnetworks)

function [Alpha_guess_subnets] = AlphaPrediction_Subnetworks(MomentaInputs, dimension, sizeAlpha, ModelGuess)

    MomentaInputs(end+1,:) = ModelGuess; % ModelGuess added as input

    Alpha_guess_subnets = zeros(1,length(sizeAlpha)); % Initialization of array for Alpha guess

    % Main network: divide Alpha values in 4 main categories (A, B, C, D)

    cd('Subnetworks_method')

    cd(['Networks_' num2str(dimension) 'D'])

    load(['NetworkMainInf' num2str(dimension) 'D.mat'])
    x = MomentaInputs;
    x(isnan(x)) = 0;
    y = netAlpha(x);
    yind = vec2ind(y);

    idxA = find(yind == 1); % Category A [0.05:0.50]
    idxB = find(yind == 2); % Category B [0.55:1.00]
    idxC = find(yind == 3); % Category C [1.05:1.50]
    idxD = find(yind == 4); % Category D [1.55:2.00]

    rangeA = [0.05,0.50]; % Alpha range of category A 
    rangeB = [0.55,1.00]; % Alpha range of category B
    rangeC = [1.05,1.50]; % Alpha range of category C
    rangeD = [1.55,2.00]; % Alpha range of category D

    cat = ['A', 'B', 'C', 'D'];

    % Refine the prediction of alpha based on the category A, B, C, D (corresponding to rangeA, rangeB, rangeC, rangeD)
    
    for c = 1:length(cat)

        load(['SubNetworkInf' cat(c) num2str(dimension) 'D.mat'])

        % Select trajectories idx belonging to trajectories of category A, B, C or D
        id = eval(['idx' cat(c)]);

        % Test the network
        x = MomentaInputs(:,id);
        x(isnan(x)) = 0;
        y = netAlphasub(x);
        yind = vec2ind(y);
        
        % For each category assign alpha as mean value of one of the five subcategory defined below (rangeSubCat)
        for k = 1:5

            rangeCat = eval(['range' cat(c)]); % Read category A, B, C or D
            rangeSubCat = reshape(linspace(rangeCat(1),rangeCat(2),10),2,5)'; % Each category is divided in 5 subcategory, in which the alpha values differ by 0.05

            % Prediction of alpha as mean of the corresponding range of the subcategory
            Alpha_guess_subnets(1,id(yind == k)) = mean(rangeSubCat(k,:)); %

        end

    end

    cd ..
    cd ..

end