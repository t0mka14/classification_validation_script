clear all, close all, clc
load data
%warning('off','all')
% Here define FALSE-TRUE column vector as a label of the two groups
y(1:25)=0;
y(26:49)=1;
y = logical(y)';
BSENZ = 0;
BSPEC = 0;
BACC = 0;
bs = 0;
v = [1 2 3 4 5 6 7 8 9 10]; %here create vector from 0..number of your features
%% This section runs the regression and clasification and leave one out validation
% TADY JE VEKTOR DAT KTERY PREVEDES DO PROMENNE x
for j = 1:length(v)
    s = nchoosek(v,j); %here set how many features you want to use
    c = size(s);   
    for i = 1:c(1)    
        x = data(:,s(i,:)); %vsechny prvky v sloupci 5 6 7
        b = glmfit(x,y,'normal');
        p = glmval(b,x,'logit');
        [ALLPD_X,ALLPD_Y,T,AUC] = perfcurve(y,p,'true');
        [ACC, SENZ, SPEC, TP,TN] = loo(x,y);
        %fprintf('AUC\n')
        %AUC
        if ACC > BACC
            BACC = ACC
            BSENZ = SENZ;
            BSPEC = SPEC;
            bs = s(i,:)
        end
    end
end
disp('The best combination of features is:')
disp(bs)
disp('With:')
fprintf('\t ACC: %f \n',BACC);
fprintf('\t SENZ: %f \n',BSENZ);
fprintf('\t SPEC: %f \n',BSPEC);

%plot(ALLPD_X,ALLPD_Y,'Color',[230/255 25/255 75/255],'Linewidth',2)
% PT = (sqrt(SENZ*(-SPEC+1))+SPEC-1)/(SENZ+SPEC-1)
