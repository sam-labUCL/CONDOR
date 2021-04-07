%% This function calculate the statistical features associated to each trajectory.

function MomentaIn = calc_inputs(tr,dim,n_inp,varargin)

    if isempty(varargin)
        delta = 3; % Width of non-overlapping moving window for power spectral density statistics
    end

    % Inizialitation of the inputs array
    MomentaIn = zeros(n_inp,1);

    q = 1; % Counter of extracted inputs

    % Trajectory duration
    Tmax = length(tr)/dim;
    MomentaIn(q) = log(Tmax);
    q = q + 1;

    for dd = 1:dim

        % Prepare trajectory: normalise to std and remove mean
        xn = utility.prep_traj(tr((dd-1)*Tmax+1:dd*Tmax));

        % Displacement 
        v = xn(2:end)-xn(1:end-1);

        % Displacement: momenta and statistics
        f = v;
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+4) = [Mf; MDf; Sf; Kf; Ef]; 
        q = q + 5;
        clear f Mf MDf SDf Sf Kf Ef

        % ||Displacement||: momenta and statistics
        f = abs(v);
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef]; 
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

        % ||Displacement||, sampling every 2 time steps: momenta and statistics
        v3 = xn(3:end)-xn(1:end-2);
        f = abs(v3);
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef]; 
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

        % ||Displacement||, sampling every 3 time steps: momenta and statistics
        v4 = xn(4:end)-xn(1:end-3);
        f = abs(v4);
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef]; 
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

        % ||Displacement||, sampling every 4 time steps: momenta and statistics
        v5 = xn(5:end)-xn(1:end-4);
        f = abs(v5);
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef];
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

        % ||Displacement||, sampling every 5 time steps: momenta and statistics
        v6 = xn(6:end)-xn(1:end-5);
        f = abs(v6);
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef]; 
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

        % ||Displacement||, sampling every 6 time steps: momenta and statistics
        v7 = xn(7:end)-xn(1:end-6);
        f = abs(v7);
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef]; 
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

        % ||Displacement||, sampling every 7 time steps: momenta and statistics
        v8 = xn(8:end)-xn(1:end-7);
        f = abs(v8);
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef]; 
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

        % ||Displacement||, sampling every 8 time steps: momenta and statistics
        v9 = xn(9:end)-xn(1:end-8);
        f = abs(v9);
        [Mf, MDf, SDf, Sf, Kf] = utility.avg_stats(f);
        MomentaIn(q:q+4) = [Mf; MDf; SDf; Sf; Kf];
        q = q + 5;
        clear f Mf MDf SDf Sf Kf Ef

        % ||Displacement||, sampling every 9 time steps: momenta and statistics
        v10 = xn(10:end)-xn(1:end-9);
        f = abs(v10);
        [Mf, MDf, SDf, Sf, Kf] = utility.avg_stats(f);
        MomentaIn(q:q+4) = [Mf; MDf; SDf; Sf; Kf]; 
        q = q + 5;
        clear f Mf MDf SDf Sf Kf Ef

        % Displacement relative change
        f = v(2:end)./v(1:end-1);
        MomentaIn(q) = median(f);
        q = q + 1;
        clear f

        % Fourier Transform: momenta and statistics
        fv = abs(fftshift(fft(v)));
        f = fv(ceil(Tmax/2):end)/Tmax;
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef];
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

        % Normalized power spectral density (PSD) statistics
        f = fv(ceil(Tmax/2):end).^2/Tmax^2;
        h = ones(size(f));
        h(1:round(Tmax/4)) = -1;
        f = f.*h;
        Pf = sum(f);
        MomentaIn(q) = Pf;
        q = q + 1;
        clear f

        Pt = zeros(ceil(Tmax/delta)-1,1); 
        for m = 1:ceil(Tmax/delta)-1
            index = (m-1)*delta;
            f = v(index+(1:delta));
            fv_time = abs(fftshift(fft(f)));
            fv_time = (fv_time(ceil(delta/2):end).^2)/delta^2;
            Pt(m) = sum(fv_time);
        end
        h = ones(size(Pt));
        h(1:round(size(Pt,1)/2)) = -1;
        dPf = sum(Pt.*h);
        MomentaIn(q) = dPf;
        q = q + 1;

        f = abs(Pt);
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef];
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

        % Signal rate of variation
        LT = ischange(xn,'variance','T',20);
        LT = sum(LT == 0)/Tmax;
        MomentaIn(q) = LT;
        q = q + 1;

        % MSD statistics
        MSD = zeros(floor(Tmax/2)-1,1);
        for n = 1:floor(Tmax/2)-1
            MSD(n) = mean((xn(n+1:n:end)-xn(1:n:end-n)).^2);
        end
        time = (1:floor(Tmax/2)-1)';
        v2 = (MSD(2:end)-MSD(1:end-1))./time(1:end-1).^2;
        f = abs(v2);
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef];
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

        % Autocorrelation function of the displacement 
        rv = xcorr(v)/Tmax;
        MomentaIn(q) = sum(rv(Tmax-1+(0:delta-1)));
        q = q + 1;

        % Wavelet transform
        wt = cwt(v);

        f = abs(wt(:,1));
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef];
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

        f = abs(wt(:,2));
        [Mf, MDf, SDf, Sf, Kf, Ef] = utility.avg_stats(f);
        MomentaIn(q:q+5) = [Mf; MDf; SDf; Sf; Kf; Ef];
        q = q + 6;
        clear f Mf MDf SDf Sf Kf Ef

    end

end
