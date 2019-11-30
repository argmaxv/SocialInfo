function [GoF, Hat] = SocialInfo_Bayesian(params,J1,C,Js,Ch,Gr,nTrial)
    %global LL
    for tr=1:nTrial
        
        % Judgment range 0 to 30 years
        J_rng = [0:1:30];
        n=max(J_rng);
        
        %Prior
        J1_mu = J1(tr);
        J1_sigma = 1/(params(1)+params(2).*C(tr));
        
        %Liklihood 
        Joth_mu = Js(tr);
        Joth_tau = 1/(params(3)+params(4).*Gr(tr));
        
        % Posterior
        J2_mu =(Joth_tau*Joth_mu+J1_sigma*J1_mu)/(J1_sigma+Joth_tau);
        J2_sigma = (J1_sigma*Joth_tau)/(J1_sigma+Joth_tau);        
        
        S0 = normpdf(J_rng,J1_mu,J1_sigma); % prior belief self
        S1 = normpdf(J_rng,J2_mu,J2_sigma); % posterior belief self
        
        S0=S0/sum(S0);
        S1=S1/sum(S1);
        
        nonzeromin = 0.1e-15; % Considering that it is not possible to compute DKL for log(0) = -Inf.
        S0(find(S0<nonzeromin))=nonzeromin;
        S1(find(S1<nonzeromin))=nonzeromin;
        KL(tr) = -dot(S0,log2(S1)-log2(S0));

    end
    
    GoF=-corr(KL',Ch);
    outputeta=glmfit(KL,Ch);
    eta = outputeta(2);
    ChHat= outputeta(2)* KL'+outputeta(1);
    RS = Ch-ChHat;
    epsilon = var(RS);

    for tr=1:nTrial
        L(tr) =-(log(2*pi*epsilon)+1/epsilon*(Ch(tr)-ChHat(tr))^2)/2;
    end
    LL=sum(L);
    
    Hat.ChHat=ChHat;
    Hat.LL=LL;
    
end
