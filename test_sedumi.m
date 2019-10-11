function time_sedumi
    dims = [2, 3, 4, 5];
    datapoints = 10000;
    tolerances = [1e-2, 1e-1];
    reported_times = zeros(size(dims,2), size(tolerances,2));
    tic;
    for i = 1:size(dims,2)
        dim = dims(i);
        for j = 1:size(tolerances,2)
            options = sdpsettings('verbose',0,...
                'cachesolvers',1,'solver','sedumi',...
                'sedumi.eps', tolerances(j),'saveduals',0, 'dualize', 0);
            rand('twister',7);
            times_tol = zeros(1, datapoints);
            for pt = 1:datapoints
                % Generate random orthonormal matrix
                V = RandOrthMat(dim);
                x = rand(dim,1);
                Q = (V*diag(x))*V';
                % Build Yalmip instance
                X = sdpvar(dim,dim);
                obj = sum(sum(Q.*X,1));
                constraints = [[[1 x']; [x X]] >=0];
                for cons = 1:dim
                    constraints = [constraints, X(cons,cons)<=x(cons)];
                end
                sol = optimize(constraints,obj, options);
                times_tol(pt) = sol.solvertime;
            end
            reported_times(i,j) = sum(times_tol)/datapoints*1000;       
        end
    end
    reported_times
    toc
end

function M=RandOrthMat(n, tol)
% M = RANDORTHMAT(n)
% generates a random n x n orthogonal real matrix.
%
% M = RANDORTHMAT(n,tol)
% explicitly specifies a thresh value that measures linear dependence
% of a newly formed column with the existing columns. Defaults to 1e-6.
%
% In this version the generated matrix distribution *is* uniform over the manifold
% O(n) w.r.t. the induced R^(n^2) Lebesgue measure, at a slight computational 
% overhead (randn + normalization, as opposed to rand ). 
% 
% (c) Ofek Shilon , 2006.


    if nargin==1
	  tol=1e-6;
    end
    
    M = zeros(n); % prealloc
    
    % gram-schmidt on random column vectors
    
    vi = randn(n,1);  
    % the n-dimensional normal distribution has spherical symmetry, which implies
    % that after normalization the drawn vectors would be uniformly distributed on the
    % n-dimensional unit sphere.

    M(:,1) = vi ./ norm(vi);
    
    for i=2:n
	  nrm = 0;
	  while nrm<tol
		vi = randn(n,1);
		vi = vi -  M(:,1:i-1)  * ( M(:,1:i-1).' * vi )  ;
		nrm = norm(vi);
	  end
	  M(:,i) = vi ./ nrm;

    end %i
        
end  % RandOrthMat