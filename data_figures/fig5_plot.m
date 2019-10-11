function fig5_plot()
    dim=3;
    file = 'fig5_data_randomQs.csv';
    nbSamples = min(500000,count_lines(file))-1;  
    eigenvalues = csvread(file,0,0,[0,0,nbSamples,dim-1]);
    eigenvalues = reshape(eigenvalues.',1,[]);
    % Create figure
    figW = 300;
    figH = 200;
    fig1 = figure('Position', [400,400,figW,figH]);

    %%%%% Plot on-diagonal Q values histogram for data points
    % Create axes
    axes1 = axes('Position',...
        [0.15 0.15 0.8 0.75]);
    hold(axes1,'on');
    % Create histogram
    histogram(eigenvalues,'NumBins',100);
    box(axes1,'on');
    
    % Save generated figure
    set(gcf,'color','w');
    set(gcf, 'renderer','OpenGL');
    save_name = 'fig5_distrib_ev.fig';
    saveas(gcf,save_name);
    print(gcf,'fig5_distrib_ev','-dpng','-r300');
    close(gcf);
end

