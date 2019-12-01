
close all;
clear all;
clc;

%% Read data
load('bhv_data.mat'); % sample data

global J1 C Js Ch Gr nTrial %LL
J1 = bhv_data(:,1);                 %J1, initial judgment
J2 = bhv_data(:,2);                 %J2, second judgment
Js=bhv_data(:,3);                    %Js, social judgment
C = bhv_data(:,4);                   %C, confidence ratings, normalized to [0-1]
Dir=Js-J1;
Ch=(J2-J1).*Dir./abs(Dir);      %Changes in judgments
Gr=bhv_data(:,5);                     %Group size [0;1]=[large;small]
nTrial=length(bhv_data);        % Number of trials

%% parameter estimates

% Find the parameters that give the best fit from m x n trials of parameter estimation. 
m_sample = 100; % sample size of parameter estimates 
n_sample = 5; % trial of parameter optimization

tic
for j=1:n_sample       
    for i = 1:m_sample
        clear b allloglike

         if j==1
            J1_beta0(j,i) = rand(1)*10;
            J2_omega0(j,i) = rand(1)*10;
            Js_beta0(j,i) = rand(1)*10;
            Js_omega0(j,i) = rand(1)*10;
         else
            % Use the best parameters from the previous batch as the initial values of params for new batch 
            J1_beta0(j,i) = prams_otim(2)+rand(1);
            J2_omega0(j,i) = prams_otim(3)+rand(1);
            Js_beta0(j,i) =  prams_otim(4)+rand(1);
            Js_omega0(j,i) =  prams_otim(5)+rand(1);
            clear prams_otim
        end

        inx0 = [J1_beta0(j,i) J2_omega0(j,i) Js_beta0(j,i) Js_omega0(j,i)];
        result.inx0{i} = inx0;

        options = optimset('Display','off','MaxIter',100000,'TolFun',1e-10,'TolX',1e-10,...
            'DiffMaxChange',1e-2,'DiffMinChange',1e-4,'MaxFunEvals',100000,'LargeScale','off');
        warning off; 
        lb = [0, 0, 0, 0];   %lower bounds on parameters
        ub = [Inf, Inf, Inf, Inf]; %upper bounds on parameters

        [params, GoF] = fmincon(@SocialInfo_Bayesian, inx0, [],[],[],[],lb,ub,[],options,J1,C,Js,Ch,Gr,nTrial); 
        [GoF] = SocialInfo_Bayesian(params, J1,C,Js,Ch,Gr,nTrial);

        J1_beta(i)= params(1);
        J1_omega(i)= params(2);
        Js_beta(i) = params(3);
        Js_omega(i) = params(4);
        param_space(i,:)=[GoF, params];
        prams_otim=param_space(find(param_space==min(param_space(:,1))),:);

    end
    best_samples(j,:)=prams_otim(1,:);
end

ParamHat=mean(best_samples(find(best_samples(:,1)==min(best_samples(:,1))),2:end),1);
[GoF, Hat] = SocialInfo_Bayesian(ParamHat, J1,C,Js,Ch,Gr,nTrial);

BIC = -2*Hat.LL + length(ParamHat);
fprintf ('Loglikelihood: %f, BIC: %f \n ', Hat.LL, BIC);
ParamHat

toc