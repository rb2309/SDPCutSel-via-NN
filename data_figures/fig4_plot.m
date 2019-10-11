function fig4_plot(varargin)
    dim=3;
    file = '';
    if ~isempty(varargin)
        file = '..//neural_nets//GenDataTest3D_fig7.csv';
    else
        file = '..//neural_nets//GenDataTest3D.csv';
    end
    nbSamples = min(500000,count_lines(file))-1;    
    % skip columns in .csv file (containing eigenvectors, eigenvalues, position)
    plus = floor(dim*(dim+3)/2.0);
    X = csvread(file,0,dim*(dim+2),[0,dim*(dim+2),nbSamples,dim*(dim+1)+plus-1]);
    onDiag = X(:,1); % or X(:,4), X(:,6)
    offDiag = X(:,2); % or X(:,3), X(:,5)
    % Create figure
    figW = 500;
    figH = 250;
    fig1 = figure('Position', [400,400,figW,figH]);

    %%%%% Plot on-diagonal Q values histogram for data points
    % Create axes
    axes1 = axes('Position',[0.08 0.15 0.4 0.75]);
    hold(axes1,'on');
    % Create histogram
    histogram(onDiag,'NumBins',100);
    box(axes1,'on');
    % Set the remaining axes properties
    set(axes1,'XTickLabel',{'-1','-0.5','0','0.5','1'});
   
    %%%%% Plot off-diagonal Q values histogram for data points
    % Create axes
    axes2 = axes('Position',[0.58 0.15 0.4 0.75]);
    hold(axes2,'on');
    % Create histogram
    histogram(offDiag,'NumBins',100);
    box(axes2,'on');
    % Set the remaining axes properties
    set(axes2,'XTickLabel',{'-1','-0.5','0','0.5','1'});
    
    % Save generated figure
    set(gcf,'color','w');
    set(gcf, 'renderer','OpenGL');
    save_name = 'fig4_distrib_data.fig';
    saveas(gcf,save_name);
    print(gcf,'fig4_distrib_data','-dpng','-r300');
    close(gcf);
end

