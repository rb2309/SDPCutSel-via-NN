% Train all neural networks
function train_NNs
    % 1 is substracted from nbSamples because indexing starts at 0
    % data files too large for 1M samples
    %nbSamples = 1000000/2-1;
    nbSamples = 10000-1;
    addpath(genpath('neural_nets'))
    train_neural_net(nbSamples,2,'GenDataTest2D.csv',true);
    train_neural_net(nbSamples,3,'GenDataTest3D.csv',true);
    train_neural_net(nbSamples,4,'GenDataTest4D.csv',true);
    train_neural_net(nbSamples,5,'GenDataTest5D.csv',true);
end

% Trains a given neural network for (dim)-dimensional SDP sub-problems
% given (nbSamples) based on data from (file), either from scratch or 
% resuming from a checkpoint
function train_neural_net(nbSamples, dim, file, resumeFromCheckpoint)
    % skip (plus) columns in .csv file (containing eigenvectors, eigenvalues, position)
    plus = floor(dim*(dim+3)/2.0);
    X = csvread(file,0,dim*(dim+1),[0,dim*(dim+1),nbSamples,dim*(dim+1)+plus-1]);
    T = csvread(file,0,dim*(dim+1)+plus,[0,dim*(dim+1)+plus,nbSamples,dim*(dim+1)+plus]);
    T = T.';
    X = X.';
    % resume/warm-start training from checkpoint-saved neural net 
    if resumeFromCheckpoint
        checkpoint = "";
        load(sprintf('training_checkpoint_neural_net_%dD',dim),'-mat',checkpoint);
        net = network(checkpoint.net);
    % otherwise start from scratch
    else
        % Decide architecture (number of layers and neurons) 
        % for each neural network solving a (dim)-dimensional 
        % SDP sub-problem 
        layers=[];
        if dim==2 || dim==3
            layers = [50,50,50];
        elseif dim==4
            layers = [64,64,64];
        else % dim==5
            layers = [64,64,64,64];
        end
        % train with scaled conjugate gradient, 
        % tansig (tanh) transfer function is default
        net = feedforwardnet(layers, 'trainscg');
        % Setup Division of Data for Training, Validation, Testing
        net.divideFcn = 'dividerand';  % Divide data randomly
        net.divideMode = 'sample';  % Divide up every sample
        net.divideParam.trainRatio = 75/100;
        net.divideParam.valRatio = 15/100;
        net.divideParam.testRatio = 10/100;
        % Maximum number of epochs to train
        net.trainParam.epochs = 20000;
        % Maximum validation failures
        net.trainParam.max_fail = 100;
        % Minimum performance gradient
        %net.trainParam.min_grad = 1e-6
    end
    % train neural net on loaded data X with targets T
    % and save a checkpoint file
    [netUpdate,~] = train(net,X,T,'useParallel','no',...
                'showResources','yes',....
                'CheckpointFile',...
                sprintf('training_checkpoint_neural_net_%dD_v2.mat',dim));
    % save trained neural net
    genFunction(netUpdate, sprintf('neural_net_%dD_v2',dim));
end  
