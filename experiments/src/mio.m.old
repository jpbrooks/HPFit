% mio() uses MIP to fit a hyperplane that minimizes the q^th residual.
% Purpose: to find the best fitting hyperplance to a set of input 
% data, where best fitting means that the q^th residual is minimized.
% which can be affected by outliers.
% Two general formulations are presented.  One is based on the 
% original presented in Bertsimas and Mazumder 2014 (2.11). The other
% is an equivalent but more compact formulation developed by JC. For 
% each formulation, there are two versions: 1) a dependent variable is
% specified (like the original published version) and 2) no dependent
% variable is specified.  
% mio() is called by run_mio.R and run_cbqmio.m.
% The MIP solver is run for 60 seconds minus the warm start time and
% the best solution is reported.  Then the MIP solver is run for 
% additional time so that the total time is 3600 seconds.
%
% Main versions and options:
% formulation:
%   - mio-bm: the original MIP formulation presented in (2.11) of 
%             Bertsimas and Mazumder (2014).  
%   - mio1: an equivalent but more compact formualtion developed by
%           JC.  
%   - lqs-mio-bm: mio-bm warmstarted with LQS.  LQS is run in R.
%   - lqs-mio1: mio1 warmstarted with LQS.  LQS is run in R.
%   - alg3-mio-bm: mio-bm warmstarted with Algorithm 3 from Bertsimas
%                  and Mazumder (2014).  mio() calls algorithm3.m.
%   - alg3-mio1: mio1 warmstarted with Algorithm 3 from Bertsimas
%                and Mazumder (2014).  mio() calls algorithm3.m.
%   - cbq-mio-bm: mio-bm warmstarted with CBq created by JC.  mio()
%                 is called by run_cbqmio.m.
%   - cbq-mio1: mio1 warmstarted with CBq created by JC. mio() is 
%               is called by run_cbqmio.m.
%   - mio-bm-first: take the first feasible solution for mio-bm.
%   - mio1-first: take the first feasible solution for mio1.
% dep_var:
%   - true: a dependent variable is specified, as in ordinary 
%           regression.  A residual is measured as the absolute 
%           difference between the response value and the 
%           vertical projection of the point on the fitted hyperplane
%   - false: no dependent is specified. The intercept term is 
%            fixed to n, the original number of variables in the
%            dataset.  A residual is measured as the  
%            distance of a point to its orthogonal projection.
% q: 
%   - the order statistic used to evaluate the fit of a hyperplane. 
%     The fit is determined by the q^th largest residual.
% NOTES:
% - when a dependent variable is specified, the corresponding 
%   coefficient is set to -1.  When one is not specified, the 
%   intercept is set to n.
% - a result datafile is created containing the data file with 
%     - full path, 
%     - iteration number, 
%     - number of rows, 
%     - number of variables, 
%     - number of non-outliers, 
%     - q, 
%     - formulation, 
%     - total squared error, 
%     - MIP runtime (3600s + timelimit - 60s), 
%     - MIP status, 
%     - gamma, 
%     - best MIP bound, 
%     - number of outliers identified as one of the q smallest by 
%       best MIP feasible solution,
%     - trimmed squared error (TSE) for the q smallest residuals, TSE
%       for m_normal residuals after 1 hour, 
%     - time used by the warm start heuristic,
%     - gamma obtained by using the beta from the warm start, 
%     - TSE for m_normal residuals for warm start, 
%     - gamma obtained after 60s of MIO
%     - TSE for m_normal residuals after 60s of MIO, gamma obtained 
%       after 3600s, 
%     - TSE for m_normal after 3600s.
% - the time limit used for the solver is initially 60s minus the time 
%   to generate a warm start.  If the time is longer than 60s, then it
%   is negative.  The time limit is then set to 
%   3600s + timelimit - 60.0 so that the warmstart plus MIO has 3600s
%   total.  
% 
% INPUTS:
% - options mentioned above
% - iteration: iteration number.  Used in the output filename.
% - datafname: full path to data file.
% - lqs_beta: an initial solution generated using LQS or CBq as implemented
%             in R or CBq.  Only used for lqs-mio-bm, lqs-mio1, cbq-mio1,
%             cbq-mio-bm.
% - m_normal: number of non-outlier rows of data.  After that they are
%             outliers.
% - resloc: path to folder where output file will reside.
% - timelimit: time recorded by R for LQS or CBq as a triplet if they are 
%              used for a warmstart.  If not, provide c(0,0,0).  The time
%              is subtracted from the time allowed for the MIO.
%
% OUTPUTS:
% - beta_star: the coefficients of the best-fit hyperplane.  When 
%              dep_var=true, the first coefficient is -1 for the 
%              dependent variable, the second coefficient is the 
%              intercept, and the remaining are the coefficients for 
%              other variables.  When dep_var=false, the first 
%              coefficient is n for the intercept and the remaining
%              are the coefficients for the other variables.
% - f_beta_star: gamma, the objective function value.  The absolute
%                value of the q^th largest residual.
 

function [beta_star, f_beta_star] = mio(iteration, datafname,q, lqs_beta, m_normal, dep_var, formulation, resloc, timelimit)

X = readtable(datafname);
X = X{:,:};
[m,n] = size(X); 
if dep_var == true
    X = [X(:,1) ones(m,1) X(:,2:n)]; % add column of 1s; first column is response, second column is 1s to represent intercept, then 2:n are data
else
    X = [ones(m,1) X]; % add column of 1s at beginning for the intercept, then 2:n are data
end
[m,n] = size(X); % now n will include the intercept column and is one more than the original data


% Set up the MIP
if strcmp(formulation, "alg3-mio-bm") | strcmp(formulation, "lqs-mio-bm") | strcmp(formulation, "mio-bm-first") | strcmp(formulation, "cbq-mio-bm") | strcmp(formulation, "mio-bm")
    model.obj = [1.0; zeros(2*m,1) ; zeros(m,1); zeros(m,1); zeros(m,1); zeros(n,1)]; % gamma ; rplus/rminus; mu; mubar; z; beta
    model.lb  = [zeros(1+5*m,1); -inf(n,1)];
    if dep_var == true % dependent variable - regression; first variable is response
        model.A   = [ 
                  sparse(-ones(m,1)) speye(m) speye(m) speye(m) -speye(m) sparse(zeros(m,m)) sparse(zeros(m,n))  ; % rplus + rminus - gamma = mubar - mu, for each point i
                  sparse(zeros(m,1)) speye(m) -speye(m) sparse(zeros(m,m)) sparse(zeros(m,m)) sparse(zeros(m,m)) sparse(X) ; % rplus - rminus = y-x^T beta, for each point i
                  sparse(zeros(1,1)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(ones(1,m)) sparse(zeros(1,n)) ; % sum of zs is q
                  sparse(ones(m,1)) sparse(zeros(m,m)) sparse(zeros(m,m)) -speye(m) sparse(zeros(m,m)) sparse(zeros(m,m)) sparse(zeros(m,n))  ; % mu <= gamma, for each point i
                  sparse(zeros(1,1)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) 1.0 sparse(zeros(1,n-1))] ; % beta_1=-1
      
        model.rhs = [ zeros(m,1) ; zeros(m,1) ; q ; zeros(m,1) ; -1];
        %model.rhs = [ zeros(m,1) ; zeros(m,1) ; q ; zeros(m,1) ; ones(m,1) ; -1];
    else % no dependent variable
        model.A   = [ 
                  sparse(-ones(m,1)) speye(m) speye(m) speye(m) -speye(m) sparse(zeros(m,m)) sparse(zeros(m,n)) ; % rplus + rminus - gamma = mubar - mu, for each point i
                  sparse(zeros(m,1)) speye(m) -speye(m) sparse(zeros(m,m)) sparse(zeros(m,m)) sparse(zeros(m,m)) sparse(X) ; % rplus - rminus = -x^T beta, for each point i
                  sparse(zeros(1,1)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(ones(1,m)) sparse(zeros(1,n)) ; % sum of zs is q
                  sparse(ones(m,1)) sparse(zeros(m,m)) sparse(zeros(m,m)) -speye(m) sparse(zeros(m,m)) sparse(zeros(m,m)) sparse(zeros(m,n)) ; % mu <= gamma for each point i
                  sparse(zeros(1,1)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) 1.0 sparse(zeros(1,n-1)) ] ; % beta_0=n; n-1 because we have a column of 1s

        model.rhs = [ zeros(m,1) ; zeros(m,1) ; q ; zeros(m,1) ; n-1];
    end
    model.sense = [repmat('=', m, 1); repmat('=', m, 1); '='; repmat('>', m, 1) ; '='] ;
    model.vtype = ['C'; repmat('C', 4*m, 1) ; repmat('B', m, 1); repmat('C', n, 1) ];
    model.varnames = cellstr(['gamma' ; repmat('rplus',m, 1) + string(1:m)' ; repmat('rminus',m, 1) + string(1:m)' ; repmat('mu',m, 1) + string(1:m)' ; repmat('mubar',m, 1) + string(1:m)' ; repmat('z',m, 1) + string(1:m)' ; repmat('beta',n, 1) + string(1:n)']) 
else % formulation is MIO1
    model.obj = [1.0; zeros(m,1) ; zeros(m,1); zeros(m,1); zeros(m,1); zeros(n,1)]; % gamma ; r; eplus; eminus; z; beta
    model.lb  = [zeros(1+4*m,1); -inf(n,1)];
    if dep_var == true % dependent variable - regression; first variable is response
        model.A   = [ sparse(zeros(m,1)) sparse(zeros(m,m)) speye(m) -speye(m) sparse(zeros(m,m)) sparse(X) ; % beta^T x - y + eplus - eminus = 0, for each point i
                      sparse(-ones(m,1)) -speye(m) speye(m) speye(m) sparse(zeros(m,m)) sparse(zeros(m,n)) ; % eplus + eminus - r - gamma <= 0, for each point i
                      sparse(zeros(1,1)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(ones(1,m)) sparse(zeros(1,n)); % sum of zs is q
                      sparse(zeros(1,1)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) 1.0 sparse(zeros(1,n-1)) ] ; % beta_1=-1
        model.rhs = [ zeros(m,1) ; zeros(m,1) ; q ; -1 ];
    else
        model.A   = [ sparse(zeros(m,1)) sparse(zeros(m,m)) speye(m) -speye(m) sparse(zeros(m,m)) sparse(X) ; % beta^Tx + eplus - eminus = 0, for each point i
                      sparse(-ones(m,1)) -speye(m) speye(m) speye(m) sparse(zeros(m,m)) sparse(zeros(m,n)) ; % eplus + eminus - r - gamma <= 0, for each point i
                      sparse(zeros(1,1)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(ones(1,m)) sparse(zeros(1,n)); % sum of zs is q
                      sparse(zeros(1,1)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) sparse(zeros(1,m)) 1.0 sparse(zeros(1,n-1)) ] ; % beta_0=n
        model.rhs = [ zeros(m,1) ; zeros(m,1) ; q ; n-1 ]; % n-1 because we have a column of 1s for the intercept
    end
    model.sense = [repmat('=', m, 1); repmat('<', m, 1); '='; '='] ;
    model.vtype = ['C'; repmat('C', 3*m, 1) ; repmat('B', m, 1); repmat('C', n, 1)];
    model.varnames = cellstr(['gamma' ; repmat('r',m, 1) + string(1:m)' ; repmat('eplus',m, 1) + string(1:m)' ; repmat('eminus',m, 1) + string(1:m)' ; repmat('z',m, 1) + string(1:m)' ; repmat('beta',n, 1) + string(1:n)']) 
end
for k=1:m
    if strcmp(formulation, "alg3-mio-bm") | strcmp(formulation, "lqs-mio-bm") | strcmp(formulation, "mio-bm-first") | strcmp(formulation, "cbq-mio-bm") | strcmp(formulation, "mio-bm")
        model.sos(k).type = 1;
        model.sos(k).index = [(1+2*m+m+k) (1+2*m+k)]'; % mubar, mu
        model.sos(m+k).type = 1;
        model.sos(m+k).index = [(1+k) (1+m+k)]'; % rplus, rminus
        model.sos(2*m+k).type = 1;
        model.sos(2*m+k).index = [(1+2*m+2*m+k) (1+2*m+m+k)]'; % z, mubar;  this is a correction from the paper.
    else % MIO1
        model.sos(k).type = 1;
        model.sos(k).index = [(1+k) (1+3*m+k)]'; % r, z
    end
end
if strcmp(formulation, "alg3-mio-bm") | strcmp(formulation, "alg3-mio1")
    disp("alg 3 start")
    tStart = tic;
    [beta_start, f_beta1] = algorithm3(X, q, dep_var, "PCA"); % algorithm 3 is given by Bertsimas and Mazumder as a way to derive initial solutions
    alg3_time = toc(tStart)
    timelimit = timelimit - alg3_time % subjtract alg_3 time from solver time
    f_beta1 % print gamma from alg3
    disp("alg 3 end")
end

if strcmp(formulation, "lqs-mio-bm") | strcmp(formulation, "lqs-mio1") | strcmp(formulation, "cbq-mio1") | strcmp(formulation, "cbq-mio-bm")
    beta_start = lqs_beta; % if lqs or cbq is used as a warmstart, it is passed as lqs_beta
end

newGammaHeur = -1.0 
tsestarHeur = -1.0 
newGamma60 = -1.0 
tsestar60 = -1.0 
newGamma3600 = -1.0 
tsestar3600 = -1.0 
% calculate values for all variables based on the warm start gamma
if strcmp(formulation, "alg3-mio-bm") | strcmp(formulation, "lqs-mio-bm") | strcmp(formulation, "cbq-mio-bm") %MIO-BM
    rplus = zeros(m,1);
    rminus = zeros(m,1);
    mu = zeros(m,1);
    mubar = zeros(m,1);
    z = zeros(m,1);
    model.StartNumber = 0;
    dist = X*beta_start;
    absdist = abs(dist); 
    sortedabsdist = [(1:m)',absdist];
    sortedabsdist = sortrows(sortedabsdist,2);
    newGammaHeur = sortedabsdist(q,2)  % gamma from warmstart
    fprintf("Gamma recalculated from beta_start is %f\n", newGammaHeur);
    for i=1:m  % calculate rminus and rplus variable values
        if dist(i,1) > 0
            rminus(i,1) = dist(i,1);
        else
            rplus(i,1) = -dist(i,1);
        end
        if rplus(i,1) + rminus(i,1) - newGammaHeur > 0  % calculate mubar and mu values
            mubar(i,1) = rplus(i,1) + rminus(i,1) - newGammaHeur;
        else
            mu(i,1) = -rplus(i,1) - rminus(i,1) + newGammaHeur;
        end
    end
    for i=1:q
        z(sortedabsdist(i,1),1) = 1;  % identify which variables are selected as inliers
    end
    model.start = [newGammaHeur; rplus; rminus; mu; mubar; z; beta_start];  % specify starting solution for MIP
    if dep_var == true % get error along response direction, recall that beta_1 = -1
        tot_err = sum(dist(1:m_normal,1).*dist(1:m_normal,1));
        sorteddist = sort(dist(:,1).*dist(:,1));
        tsestarHeur = sum(sorteddist(1:m_normal)) % TSEstar is based on m_normal, which we usually don't know
        tse = sum(sorteddist(1:q)) % TSE is based on q, which we usually set to 0.5
    else % get orthogonal error
        gradLen = norm(beta_start(2:n,1)); % first coefficient is the intercept; exclude that from the gradLen calculation
        edist = abs(dist/gradLen);
        sortededist = sort(edist(:,1).*edist(:,1));
        tsestarHeur = sum(sortededist(1:m_normal)) % TSEstar is based on m_normal, which we usually don't know
        tse = sum(sortededist(1:q)) % TSE is based on q, which we usually set to 0.5
        tot_err = sum(edist(1:m_normal,1).*edist(1:m_normal,1));
    end
elseif strcmp(formulation, "alg3-mio1") | strcmp(formulation, "lqs-mio1") | strcmp(formulation, "cbq-mio1") %MIO1
    eplus = zeros(m,1);
    eminus = zeros(m,1);
    rel = zeros(m,1);
    z = zeros(m,1);
    model.StartNumber = 0;
    dist = X*beta_start;
    absdist = abs(dist);
    sortedabsdist = [(1:m)',absdist];
    sortedabsdist = sortrows(sortedabsdist,2);
    newGammaHeur = sortedabsdist(q,2)  
    fprintf("Gamma recalculated from beta_start is %f\n", newGammaHeur);
    for i=1:m % calculate eminus and eplus values
        if dist(i,1) > 0 
            eminus(i,1) = dist(i,1);
        else
            eplus(i,1) = -dist(i,1);
        end
    end
    rel = absdist;
    for i=1:q
        z(sortedabsdist(i,1),1) = 1; % specify inliers
        rel(sortedabsdist(i,1),1) = 0; 
    end
    model.start = [newGammaHeur; rel; eplus; eminus; z; beta_start];   % specify initial solution
    if dep_var == true % get error along response direction, recall that beta_1 = -1
        tot_err = sum(dist(1:m_normal,1).*dist(1:m_normal,1));
        sorteddist = sort(dist(:,1).*dist(:,1));
        tsestarHeur = sum(sorteddist(1:m_normal)) % TSE* is for m_normal, which we usually don't know
        tse = sum(sorteddist(1:q)) % TSE is for q, which is usually 0.5
    else % get orthogonal error
        gradLen = norm(beta_start(2:n,1)); % first coefficient is the intercept; exclude that from the gradLen calculation
        edist = abs(dist/gradLen);
        sortededist = sort(edist(:,1).*edist(:,1));
        tsestarHeur = sum(sortededist(1:m_normal)) % TSE* is for m_normal, which we usually don't know
        tse = sum(sortededist(1:q)) % TSE is for q, which is usually 0.5
        tot_err = sum(edist(1:m_normal,1).*edist(1:m_normal,1));
    end
end


model.modelsense = 'min';
%gurobi_write(model, 'mio.lp');
params = struct();
if strcmp(formulation, "mio-bm-first") | strcmp(formulation, "mio1-first")
    params.SolutionLimit = 1;  % take the first solution
end

%params.OutputFlag = 0;
if timelimit >= 0.0 % if the warm start took less than 60 seconds, then run MIO for the remaining time.
    params.TimeLimit = timelimit;
    params.Symmetry = 2;
    params.Threads = 14 
    
    disp("solving 60 s")
    result = gurobi(model, params);
    result.status
    if strcmp(result.status, 'OPTIMAL')  % get the optimal solution from Gurobi
        if strcmp(formulation, "alg3-mio-bm") | strcmp(formulation, "lqs-mio-bm") | strcmp(formulation, "mio-bm-first") | strcmp(formulation, "cbq-mio-bm") | strcmp(formulation, "mio-bm")
            beta_star = result.x((1+5*m+1):(1+5*m+n),1);
            z = result.x(1+4*m+1:1+4*m+m,1);
        else % MIO1
            beta_star = result.x((1+4*m+1):(1+4*m+n),1);
            z = result.x(1+3*m+1:1+3*m+m,1);
        end
    else  % get the best known solution
        fprintf("Using incumbent solution\n")
        if strcmp(formulation, "alg3-mio-bm") | strcmp(formulation, "lqs-mio-bm") | strcmp(formulation, "mio-bm-first") | strcmp(formulation, "cbq-mio-bm") | strcmp(formulation, "mio-bm")
            beta_star = result.pool(1).xn((1+5*m+1):(1+5*m+n),1);
            z = result.pool(1).xn(1+4*m+1:1+4*m+m,1);
        else % MIO1
            beta_star = result.pool(1).xn((1+4*m+1):(1+4*m+n),1);
            z = result.pool(1).xn(1+3*m+1:1+3*m+m,1);
        end
        %else
        %    fprintf("No solution available\n")
        %    beta_star = zeros(n,1);
        %    %return
        %end
    end
    f_beta_star = result.objval;
    %qth_residuals = [f_beta1 f_beta_star];
    
    if strcmp(formulation, "mio-bm-first") | strcmp(formulation, "mio1-first")
        output_runtime = result.runtime;
    end
    
    num_outliers_in_q = sum(z((m_normal+1):m,1)) % number of outliers identified as inliers
    
    % get sum of squared error on non outliers
    dist = abs(X*beta_star); 
    if dep_var == true % get error along response direction, recall that beta_1 = -1
        tot_err = sum(dist(1:m_normal,1).*dist(1:m_normal,1));
        sorteddist = sort(dist(:,1).*dist(:,1));
        tsestar60 = sum(sorteddist(1:m_normal)) % TSE* is based on m_normal, which we usually don't know
        tse = sum(sorteddist(1:q)) % TSE is based on q, which we usually set to 0.5
    else % get orthogonal error
        gradLen = norm(beta_star(2:n,1)); % first coefficient is the intercept; exclude that from the gradLen calculation
        edist = abs(dist/gradLen);
        sortededist = sort(edist(:,1).*edist(:,1));
        tsestar60 = sum(sortededist(1:m_normal))% TSE* is based on m_normal, which we usually don't know
        tse = sum(sortededist(1:q))% TSE is based on q, which we usually set to 0.5
        tot_err = sum(edist(1:m_normal,1).*edist(1:m_normal,1));
    end
    
    % get 60s solution to start 
    if strcmp(formulation, "mio-bm") | strcmp(formulation, "alg3-mio-bm") | strcmp(formulation, "lqs-mio-bm") | strcmp(formulation, "cbq-mio-bm") | strcmp(formulation, "mio-bm-first") %MIO-BM
        rplus = zeros(m,1);
        rminus = zeros(m,1);
        mu = zeros(m,1);
        mubar = zeros(m,1);
        z = zeros(m,1);
        model.StartNumber = 0;
        dist = X*beta_star;
        absdist = abs(dist);
        sortedabsdist = [(1:m)',absdist];
        sortedabsdist = sortrows(sortedabsdist,2);
        newGamma60 = sortedabsdist(q,2)
        fprintf("Gamma recalculated from beta_start is %f\n", newGamma60);
        for i=1:m % calculate rminus and rplus
            if dist(i,1) > 0
                rminus(i,1) = dist(i,1);
            else
                rplus(i,1) = -dist(i,1);
            end
            if rplus(i,1) + rminus(i,1) - newGamma60 > 0 % calculate mubar and mu
                mubar(i,1) = rplus(i,1) + rminus(i,1) - newGamma60;
            else
                mu(i,1) = -rplus(i,1) - rminus(i,1) + newGamma60;
            end
        end
        for i=1:q
            z(sortedabsdist(i,1),1) = 1;  % specify inliers
        end
        model.start = [newGamma60; rplus; rminus; mu; mubar; z; beta_star];  
    elseif strcmp(formulation, "mio1") | strcmp(formulation, "alg3-mio1") | strcmp(formulation, "lqs-mio1") | strcmp(formulation, "cbq-mio1") | strcmp(formulation, "mio1-first") %MIO1
        eplus = zeros(m,1);
        eminus = zeros(m,1);
        rel = zeros(m,1);
        z = zeros(m,1);
        model.StartNumber = 0;
        dist = X*beta_star;
        absdist = abs(dist);
        sortedabsdist = [(1:m)',absdist];
        sortedabsdist = sortrows(sortedabsdist,2);
        newGamma60 = sortedabsdist(q,2)  
        fprintf("Gamma recalculated from beta_start is %f\n", newGamma60);
        for i=1:m % calculate eminus, eplus
            if dist(i,1) > 0
                eminus(i,1) = dist(i,1);
            else
                eplus(i,1) = -dist(i,1);
            end
        end
        rel = absdist;
        for i=1:q
            z(sortedabsdist(i,1),1) = 1;  % specify inliers
            rel(sortedabsdist(i,1),1) = 0;
        end
        model.start = [newGamma60; rel; eplus; eminus; z; beta_star];  % set initial solution
    end
end

disp("solving 60 m")
params.TimeLimit = 3600.0 + timelimit - 60.0;  %timelimit has 60s for first MIO built in, minus heuristic; timelimit may be negative if heuristic took longer than 60s
result = gurobi(model, params);

if strcmp(result.status, 'OPTIMAL')  % get optimal solution
    if strcmp(formulation, "alg3-mio-bm") | strcmp(formulation, "lqs-mio-bm") | strcmp(formulation, "mio-bm-first") | strcmp(formulation, "cbq-mio-bm") | strcmp(formulation, "mio-bm")
        beta_star = result.x((1+5*m+1):(1+5*m+n),1);
        z = result.x(1+4*m+1:1+4*m+m,1);
    else % MIO1
        beta_star = result.x((1+4*m+1):(1+4*m+n),1);
        z = result.x(1+3*m+1:1+3*m+m,1);
    end
else  % get best solution
    %if result.mipgap ~= Inf
    fprintf("Using incumbent solution\n")
    if strcmp(formulation, "alg3-mio-bm") | strcmp(formulation, "lqs-mio-bm") | strcmp(formulation, "mio-bm-first") | strcmp(formulation, "cbq-mio-bm") | strcmp(formulation, "mio-bm")
        beta_star = result.pool(1).xn((1+5*m+1):(1+5*m+n),1);
        z = result.pool(1).xn(1+4*m+1:1+4*m+m,1);
    else % MIO1
        beta_star = result.pool(1).xn((1+4*m+1):(1+4*m+n),1);
        z = result.pool(1).xn(1+3*m+1:1+3*m+m,1);
    end
    %else
    %    fprintf("No solution available\n")
    %    beta_star = zeros(n,1);
    %    %return
    %end
end
%qth_residuals = [f_beta1 f_beta_star];

num_outliers_in_q = sum(z((m_normal+1):m,1)) % number of outliers identified as inliers
f_beta_star = result.objval;

% get sum of squared error on non outliers
dist = abs(X*beta_star); 
if dep_var == true % get error along response direction, recall that beta_1 = -1
    tot_err = sum(dist(1:m_normal,1).*dist(1:m_normal,1));
    sorteddist = sort(dist(:,1).*dist(:,1));
    tsestar3600 = sum(sorteddist(1:m_normal))% TSE* is based on m_normal, which we usually don't know
    tse = sum(sorteddist(1:q))% TSE is based on q, which we usually set to 0.5
else % get orthogonal error
    gradLen = norm(beta_star(2:n,1)); % first coefficient is the intercept; exclude that from the gradLen calculation
    edist = abs(dist/gradLen);
    sortededist = sort(edist(:,1).*edist(:,1));
    tsestar3600 = sum(sortededist(1:m_normal))% TSE* is based on m_normal, which we usually don't know
    tse = sum(sortededist(1:q))% TSE is based on q, which we usually set to 0.5
    tot_err = sum(edist(1:m_normal,1).*edist(1:m_normal,1));
end

% calculate gamma for 3600 solution

if strcmp(formulation, "mio-bm") | strcmp(formulation, "alg3-mio-bm") | strcmp(formulation, "lqs-mio-bm") | strcmp(formulation, "cbq-mio-bm") %MIO-BM
    rplus = zeros(m,1);
    rminus = zeros(m,1);
    mu = zeros(m,1);
    mubar = zeros(m,1);
    z = zeros(m,1);
    model.StartNumber = 0;
    dist = X*beta_star;
    absdist = abs(dist);
    sortedabsdist = [(1:m)',absdist];
    sortedabsdist = sortrows(sortedabsdist,2);
    newGamma3600 = sortedabsdist(q,2)
    fprintf("Gamma recalculated from beta_start is %f\n", newGamma3600);
elseif strcmp(formulation, "mio1") | strcmp(formulation, "alg3-mio1") | strcmp(formulation, "lqs-mio1") | strcmp(formulation, "cbq-mio1") %MIO1
    eplus = zeros(m,1);
    eminus = zeros(m,1);
    rel = zeros(m,1);
    z = zeros(m,1);
    model.StartNumber = 0;
    dist = X*beta_star;
    absdist = abs(dist);
    sortedabsdist = [(1:m)',absdist];
    sortedabsdist = sortrows(sortedabsdist,2);
    newGamma3600 = sortedabsdist(q,2)  
    fprintf("Gamma recalculated from beta_start is %f\n", newGamma3600);
end

out_fname = strcat(resloc, "/", formulation,"i",int2str(iteration), ".csv")
disp(out_fname)
out_file = fopen(out_fname, "w");

if ~(strcmp(formulation, "mio-bm-first") | strcmp(formulation, "mio1-first"))
    output_runtime = result.runtime;
end

%disp(result.status)
beta_star
 
newGamma60

% - a result datafile is created containing the data file with full 
%   path, iteration number, number of rows, number of variables, 
%   number of non-outliers, q, formulation, total squared error, 
%   MIP runtime, MIP status, gamma, best MIP bound, number of outliers
%   identified as one of the q smallest by best MIP feasible solution,
%   trimmed squared error (TSE) for the q smallest residuals, TSE for 
%   m_normal residuals after 1 hour, time limit used for the solver,
%   gamma obtained by using the beta from the warm start, TSE for 
%   m_normal residuals for warm start, gamma obtained after 60s of MIO
%   TSE for m_normal after 60s of MIO, gamma obtained after 3600s, TSE
%   for m_normal after 3600s.

fprintf(out_file, "%s,%d,%d,%d,%d,%d,%s,%f,%f,%s,%f,%f,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", datafname, iteration, m, n-1, m_normal, q, formulation, tot_err, output_runtime, result.status, f_beta_star, result.objbound, num_outliers_in_q,tse,tsestar3600,timelimit,newGammaHeur,tsestarHeur,newGamma60,tsestar60,newGamma3600,tsestar3600);

fclose(out_file);

return
        
end
