function fig9_10_plot
    cutsRounds = 20;
    cols = 15;
    data_file = 'fig9_10_data.csv';
    fid = fopen(data_file);
    nb_instances = floor(count_lines(data_file)/(3+2*(cutsRounds+3)));   
    for figNb=1:nb_instances
        row1 = (3+2*(cutsRounds+3))*(figNb-1);
        fseek(fid,0,'bof');
        a = textscan(fid,'%s','Delimiter',',','headerlines',row1);
        filename = a{1}{1};
        b = csvread(data_file, row1+2,0,[row1+2, 0, row1+2 ,2]);
        gaps = csvread(data_file, row1+5,0,...
            [row1+5, 0, row1+5+cutsRounds ,cols])*100;
        row1 = row1+7+cutsRounds+1;
        times = csvread(data_file, row1,0,...
            [row1, 0, row1+cutsRounds ,cols]);
        % To plot M+S^E_3 as a straight line
        times(cutsRounds+1,16)=10000;
        times(1:cutsRounds,16)=0.001;
        % Plot across cuts rounds and across time
        plotIters(gaps,b,figNb,filename)
        plotTime(gaps, times,b,figNb,filename)
    end    
end

function plotTime(gaps, times,b,figNb,filename)
    rdnText = 'Random sel.';
    NNtext = 'Optimality sel. (via estimated $\hat{\mathcal{I}}_X(\rho)$)';
    EXtext = 'Optimality sel. (via exact $\mathcal{I}_X(\rho)$)';
    Feastext = 'Feasibility sel. (via $\lambda_{\min}(\rho)$)';
    Combtext = 'Combined (opt.+feas.) sel. (via $\mathcal{C}(\rho)$)';
    Boundtext = '$\mathcal{M}+\mathcal{S}^{E}_3$ bound';
    Densetext = 'All violated (up to $N$) dense eigencuts';
    figW = 450;
    figH = 350;
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';    
    % Create axes
    axes1 = axes('Position',[0.16 0.15 0.8 0.75]);  
    hold(axes1,'on');  
    % Create multiple lines using matrix input to plot
    plot1 = semilogx (times(:,4:13),gaps(:,4:13),'LineWidth',2,'LineStyle',':','Color',[0.5 0.5 0.5]);
    set(plot1(1),'DisplayName',rdnText);
    semilogx(times(:,16),gaps(:,16),'DisplayName',Boundtext,'LineWidth',2,'Color',[0 0.5 0],'LineStyle','-');
    semilogx(times(:,15),gaps(:,15),'DisplayName',Densetext,'LineWidth',2,'Marker','square','Color',[0.87 0.5 0],'LineStyle','-');
    semilogx(times(:,14),gaps(:,14),'DisplayName',EXtext,'LineWidth',6,'LineStyle',':','Color',[0.301960784313725 0.749019607843137 0.929411764705882]);
    semilogx(times(:,1),gaps(:,1),'DisplayName',NNtext,'LineWidth',2,'Color',[0 0 1],'LineStyle','-','Marker','o', 'MarkerSize',8);
    semilogx(times(:,2),gaps(:,2),'DisplayName',Feastext,'MarkerSize',6,'Marker','square','LineWidth',2,'Color',[1 0 0],'LineStyle','-');
    semilogx(times(:,3),gaps(:,3),'DisplayName',Combtext,'LineWidth',3,'Color',[0 0 0],'LineStyle','-','Marker','none', 'MarkerSize',8);    
    set(gca,'xscale','log');
    xlim(axes1,[min(min(times(2:end,1:15))) max(max(times(:,1:15)))]);
    ylim(axes1,[0 100]);
    ytickformat(axes1,'percentage');
    box(axes1,'on');
    
    % Create ylabel
    ylabel('\% of $\mathcal{M}$ to $(\mathcal{M}+\mathcal{S})$ gap closed',...
        'FontSize',13,...
        'Interpreter','latex');
    % Create xlabel
    xlabel('Log times (s)','FontSize',13,...
        'Interpreter','latex');
    % Create title
    title([filename,' ($N$=' , num2str(b(1)),' vars, ' , num2str(b(2)), '\% dense)'],...
        'FontSize',14,'FontWeight','normal',...
        'Interpreter','latex');
    box(axes1,'on');
    set(axes1,'XGrid','on','XMinorGrid','on');
    set(gcf, 'renderer','OpenGL');
    set(gcf,'color','w');
    saveas(gcf,sprintf('fig10_%d.fig',figNb));
    print(gcf,sprintf('fig10_%d',figNb),'-dpng','-r300');   
    if figNb==1
        % save common legend
        h = findobj(gca,'Type','line');
        legend1 = legend([h(3),h(4),h(2),h(1),h(16),h(5),h(6)]);
        set(legend1,'Location','southwest','Interpreter','latex','FontSize',16);
        legend1.EdgeColor = 'white';
        saveLegendToImage(fig1, legend1, 9);
    end
    close(gcf);
end

function plotIters(gaps,b,figNb,filename)
    rdnText = 'Random sel.';
    NNtext = 'Optimality sel. (estimated $\hat{\mathcal{I}}_X(\rho)$)';
    EXtext = 'Optimality sel. (exact $\mathcal{I}_X(\rho)$)';
    Feastext = 'Feasibility sel. ($\lambda_{\min}(\rho)$)';
    Combtext = 'Combined sel. ($\mathcal{C}(\rho)$)';
    Boundtext = '$\mathcal{M}+\mathcal{S}^{E}_3$ bound';
    Densetext = 'All violated (up to $N$) dense eigencuts';
    figW = 450;
    figH = 350;
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';   
    % Create axes
    axes1 = axes('Position',[0.16 0.16 0.78 0.75]);  
    hold(axes1,'on');
   
    % Create multiple lines using matrix input to plot
    plot1 = plot(gaps(:,4:13),'LineWidth',2,'LineStyle',':','Color',[0.5 0.5 0.5]);
    set(plot1(1),'DisplayName',rdnText);
    plot(gaps(:,16),'DisplayName',Boundtext,'LineWidth',2,'Color',[0 0.5 0],'LineStyle','-');
    plot(gaps(:,15),'DisplayName',Densetext,'LineWidth',2,'Marker','square','Color',[0.87 0.5 0],'LineStyle','-');
    plot(gaps(:,14),'DisplayName',EXtext,'LineWidth',6,'LineStyle',':','Color',[0.301960784313725 0.749019607843137 0.929411764705882]);
    plot(gaps(:,1),'DisplayName',NNtext,'LineWidth',2,'Color',[0 0 1],'LineStyle','-','Marker','o', 'MarkerSize',8);
    plot(gaps(:,2),'DisplayName',Feastext,'MarkerSize',6,'Marker','square','LineWidth',2,'Color',[1 0 0],'LineStyle','-');
    plot(gaps(:,3),'DisplayName',Combtext,'LineWidth',3,'Color',[0 0 0],'LineStyle','-','Marker','none', 'MarkerSize',8);      
    xlim(axes1,[1 20]);
    ylim(axes1,[0 100]);
    ytickformat(axes1,'percentage');
    box(axes1,'on');
    
    % Create ylabel
    ylabel('\% of $\mathcal{M}$ to $(\mathcal{M}+\mathcal{S})$ gap closed',...
        'FontSize',13,...
        'Interpreter','latex');
    % Create xlabel
    xtitle = xlabel(['Cut rounds (selecting max $5\%\cdot| \mathcal{P}^{E}_3|=$ ', sprintf('%d cuts/round)', b(3))],'FontSize',13,...
        'Interpreter','latex');
    set(xtitle,'Position', get(xtitle,'Position') + [-1 -1 0]);
    % Create title
    title([filename,' ($N$=' , num2str(b(1)),' vars, ' , num2str(b(2)), '\% dense)'],...
        'FontSize',14,'FontWeight','normal',...
        'Interpreter','latex');

    box(axes1,'on');
    set(axes1,'XGrid','on','XMinorGrid','on');    
    set(gcf, 'renderer','OpenGL');
    set(gcf,'color','w');
    saveas(gcf,sprintf('fig9_%d.fig',figNb));
    print(gcf,sprintf('fig9_%d',figNb),'-dpng','-r300');
    if figNb==1
        % save common legend
        h = findobj(gca,'Type','line');
        legend1 = legend([h(3),h(4),h(2),h(1),h(16),h(5),h(6)]);
        set(legend1,'Location','southwest','Interpreter','latex','FontSize',16);
        legend1.EdgeColor = 'white';
        saveLegendToImage(fig1, legend1, 9);
    end
    close(gcf);
end

function saveLegendToImage(figHandle, legHandle, figNb)
    warning('off','all');
    %make all contents in figure invisible
    allLineHandles = findall(figHandle, 'type', 'line');
    for i = 1:length(allLineHandles)
        allLineHandles(i).XData = NaN; %ignore warnings
    end
    %make axes invisible
    axis off
    %move legend to lower left corner of figure window
    legHandle.Units = 'pixels';
    boxLineWidth = legHandle.LineWidth;
    %save isn't accurate and would swallow part of the box without factors
    legHandle.Position = [6 * boxLineWidth, 6 * boxLineWidth+7, ...
        legHandle.Position(3), legHandle.Position(4)];
    legLocPixels = legHandle.Position;
    %make figure window fit legend
    figHandle.Units = 'pixels';
    figHandle.InnerPosition = [1, 1, legLocPixels(3) + 12 * boxLineWidth, ...
        legLocPixels(4) + 12 * boxLineWidth];
    %save legend
    saveas(figHandle,sprintf('fig%d_legends.fig',figNb));
    print(figHandle,sprintf('fig%d_legends',figNb),'-dpng','-r300');
    warning('on','all');
    close(gcf);
end
