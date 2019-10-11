function fig15_plot
    cutsRounds = 10;
    cols = 6;
    data_file = 'fig15_data.csv';
    fid = fopen(data_file);
    nb_instances = floor(count_lines(data_file)/(4+(cutsRounds+1)));   
    for figNb=1:nb_instances
        row1 = (cutsRounds+1+4)*(figNb-1)+2;
        fseek(fid,0,'bof');
        a = textscan(fid,'%s','Delimiter',',','headerlines',row1-2);
        filename = a{1}{1};
        b = csvread(data_file, row1,0,[row1, 0, row1 ,3]);
        gaps = csvread(data_file, row1+2,0,...
            [row1+2, 0, row1+2+cutsRounds ,cols])*100;
        plotIters(gaps,b,figNb,filename)
    end    
end


function plotIters(gaps,b,figNb,filename)
    rdnText = 'Random sel.';
    Feastext = 'Feasibility sel. ($\lambda_{\min}(\rho)$)';
    Combtext = 'Combined sel. ($\mathcal{C}^{(0)}(\rho)$)';
    figW = 450;
    figH = 350;
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';   
    % Create axes
    axes1 = axes('Position',[0.16 0.16 0.78 0.75]);  
    hold(axes1,'on');
   
    % Create multiple lines using matrix input to plot
    plot1 = plot(gaps(:,3:7),'LineWidth',2,'LineStyle',':','Color',[0.5 0.5 0.5]);
    set(plot1(1),'DisplayName',rdnText);
    plot(gaps(:,1),'DisplayName',Feastext,'MarkerSize',6,'Marker','square','LineWidth',2,'Color',[1 0 0],'LineStyle','-');
    plot(gaps(:,2),'DisplayName',Combtext,'MarkerSize',6,'Marker','square','LineWidth',2,'Color',[0 0 0],'LineStyle','-');      
    xlim(axes1,[1 10]);
    ylim(axes1,[0 80]);
    ytickformat(axes1,'percentage');
    box(axes1,'on');
    
    % Create ylabel
    ylabel('\% of $\mathcal{M}$ to optimality gap closed',...
        'FontSize',13,...
        'Interpreter','latex');
    % Create xlabel
    xtitle = xlabel(['Cut rounds (selecting max $5\%\cdot| \mathcal{P}^{E_0}_3|=$ ', sprintf('%d cuts/round)', b(4))],'FontSize',13,...
        'Interpreter','latex');
    set(xtitle,'Position', get(xtitle,'Position') + [-0.5 -1 0]);
    % Create title
    bigtitle = title([filename,' (' , num2str(b(1)), ' vars, ', num2str(b(2)), ' constraints, ', num2str(b(3)), '% dense)'],...
        'FontSize',12,'FontWeight','normal',...
        'Interpreter','none');
    set(bigtitle,'Position', get(bigtitle,'Position') + [0 2 0]);

    box(axes1,'on');
    set(axes1,'XGrid','on','XMinorGrid','on');    
    % h = findobj(gca,'Type','line');
    % legend1 = legend([h(3),h(4),h(2),h(14),h(1)]);
    % set(legend1,'Interpreter','latex','FontSize',12,'Location','best');
    set(gcf, 'renderer','OpenGL');
    set(gcf,'color','w');
    saveas(gcf,sprintf('fig15_%d.fig',figNb));
    print(gcf,sprintf('fig15_%d',figNb),'-dpng','-r300');
    if figNb==1
        % save common legend
        h = findobj(gca,'Type','line');
        legend1 = legend([h(1),h(2),h(7)]);
        set(legend1,'Location','southwest','Interpreter','latex','FontSize',16);
        legend1.EdgeColor = 'white';
        saveLegendToImage(fig1, legend1, 15);
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
