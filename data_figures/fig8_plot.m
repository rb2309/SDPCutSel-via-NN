function fig8_plot
    % skip first 6 lines of the csv and read only the data on cuts
    % skip first column (number of the cuts round)
    M = csvread('fig8_data.csv', 6,1);
    nbRounds = 4;
    nbTriples = length(M)/nbRounds;
    figure1 = figure;
    for r=0:nbRounds-1
       Ms = M(r*nbTriples+1:(r+1)*nbTriples,:);   
       Ms1 = sortrows(Ms,1);
       Ms1sel = Ms1(Ms1(:,2)>0,[1,5]);
       Ms1not = Ms1(Ms1(:,2)==0,[1,5]);
       createaxes1(Ms1not(:,1),Ms1not(:,2),Ms1sel(:,1),Ms1sel(:,2),nbTriples,r);

       % show only top 400 sub-problems for sorted columns 2-3
       showCount = 400;

       Ms2 = sortrows(Ms,5,'descend');
       Ms2 = Ms2(1:showCount,:);
       Ms2(:,1)=1:showCount;
       Ms2sel = Ms2(Ms2(:,2)>0,[1,5]);
       Ms2not = Ms2(Ms2(:,2)==0,[1,5]);
       createaxes2(Ms2not(:,1),Ms2not(:,2),Ms2sel(:,1),Ms2sel(:,2),showCount,r);

       Ms2 = sortrows(Ms,4,'descend');
       Ms2 = Ms2(1:showCount,:);
       Ms2(:,1)=1:showCount;
       Ms2sel = Ms2(Ms2(:,2)>0,[1,5]);
       Ms2not = Ms2(Ms2(:,2)==0,[1,5]);
       createaxes3(Ms2not(:,1),Ms2not(:,2),Ms2sel(:,1),Ms2sel(:,2),showCount,r);
    end    
    % Save generated figure
    set(gcf, 'renderer','OpenGL');
    set(gcf,'color','w');
    set(gcf, 'Position', [10 10 800 900]);
    saveas(gcf,'fig8_selection_example.fig');
    print(gcf,'fig8_selection_example','-dpng','-r300');
    close(gcf);
end

% Plot first column (all rho sub-problems unordered)
function createaxes1(X1, Y1, X2, Y2,nbTriples,r)
    axes1 = axes('Position',[0.1 0.8-r*0.23 0.25 0.16]);
    hold(axes1,'on');
    % Create stems
    stem(X1,Y1,'MarkerSize',0.001,'Color',[0.3843 0.6157 0.9882]);
    stem(X2,Y2,'MarkerSize',5,'Color',[0 0 0]);
    xlim(axes1,[0 nbTriples]);
    ylim(axes1,[-15 40]);    
    grid(axes1,'on');
    % cuts round r
    if r==3
        text('Parent',axes1,'FontSize',18,'Rotation',90,'Interpreter','latex',...
            'String','$\mathcal{I}_X(\rho)$',...
            'Position',[-247.542857142857 120.430555555556 0]);
        xlabel({'All $\rho$ sub-problems';'unordered'},'Interpreter','latex','FontSize',15) ;
    end
end

% Plot second column (top sub-problems ordered by exact obj. improvement)
function createaxes2(X1, Y1, X2, Y2,nbTriples,r)
    axes1 = axes('Position',[0.40 0.8-r*0.23 0.25 0.16]);
    hold(axes1,'on');
    % Create stems
    stem(X1,Y1,'MarkerSize',0.001,'Color',[0.3843 0.6157 0.9882]);
    stem(X2,Y2,'MarkerSize',0.0001,'LineWidth',1,'Color',[0 0 0]);
    xlim(axes1,[0 nbTriples]);
    % cuts round r
    if r==0
        ylim(axes1,[-15 40]);  
    else
        ylim(axes1,[min(min(Y1),min(Y2)), max(max(Y1),max(Y2))]);
    end
    hold(axes1,'off');
    th = title({['Round ' num2str(r) ' of $\mathcal{S}_3$ cuts']},'FontSize',15, 'Interpreter','latex');
    titlePos = get( th , 'position');
    titlePos(2) = titlePos(2)*(1.03+(0.06)*r);
    set( th , 'position' , titlePos);
    hold(axes1,'on');
    grid(axes1,'on');
    if r==3
       xlabel({'Top 400 $\rho$ sub-problems';'ordered by $\mathcal{I}_X(\rho)$'},'Interpreter','latex','FontSize',14); 
    end
end

% Plot third column (top sub-problems ordered by estimated obj. improvement)
function createaxes3(X1, Y1, X2, Y2,nbTriples,r)
    axes1 = axes('Position',[0.70 0.8-r*0.23 0.25 0.16]);
    hold(axes1,'on');
    % Create stems
    stem(X1,Y1,'MarkerSize',0.001,'Color',[0.3843 0.6157 0.9882]);
    stem(X2,Y2,'MarkerSize',0.0001,'LineWidth',1,'Color',[0 0 0]);
    xlim(axes1,[0 nbTriples]);
    % cuts round r
    if r==0
        ylim(axes1,[-15 40]);  
    else
        ylim(axes1,[min(min(Y1),min(Y2)), max(max(Y1),max(Y2))]);
    end
    grid(axes1,'on');
    if r==3
       xlabel({'Top 400 $\rho$ sub-problems';'ordered by $\hat{\mathcal{I}}_X(\rho)$'},'Interpreter','latex','FontSize',14); 
    end
end