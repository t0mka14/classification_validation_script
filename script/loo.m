function [ACC, SENZ, SPEC, TP, TN, P] = loo(x,y);

for j = 1:length(y)
    
    train_x = x;
    train_x(j,:) = [];
    train_y = y;
    train_y(j,:) = [];
    test_x = x(j,:);
    test_y = y(j,:);
    
    modelparams = glmfit(train_x,train_y,'binomial','link','logit');
    modelfitLR(j) = glmval(modelparams,test_x,'logit') >= 0.5;
    P(j) = glmval(modelparams,test_x,'logit');

end
%     ACC = sum(modelfitLR==y)/length(y)*100
% SEN = sum(modelfitLR==1)/length(y)*100
C=confusionmat(modelfitLR,y);
ACC=(C(1,1)+C(2,2))/sum(C(:));
SENZ=C(1,1)/(C(1,1)+C(1,2));
SPEC=C(2,2)/(C(2,1)+C(2,2));
TP = C(1,1);
TN = C(1,2);