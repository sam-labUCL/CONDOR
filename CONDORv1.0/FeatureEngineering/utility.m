%% Class with utility functions to extract statistical features

classdef utility
    methods(Static)
        
        % Preprocessing of the trajectory (normalise to std and subtract mean)
        function xn = prep_traj(traj)
            
            xn = traj-mean(traj);
            Delta_x = xn(2:end)-xn(1:end-1);
            normalization = std(Delta_x);
            normalization(normalization==0) = 1;
            Delta_x = Delta_x./normalization;
            xn = [0 cumsum(Delta_x)];
            xn = xn-mean(xn);
            
        end
        
        % Compute statistical momenta and quantities
        function [Mf, MDf, SDf, Sf, Kf, Ef, Rf] = avg_stats(f)
            
            Mf = mean(f);
            MDf = median(f);
            SDf = std(f);
            Sf = skewness(f);
            Kf = kurtosis(f);
            if length(f) > 2
                Ef = approximateEntropy(f);
            else
                Ef = 0;
            end
            Rf = rms(f);
            
        end
        
    end
end