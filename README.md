# SocialInfo
Predicting changes in judgments while integrating the judgments of others (Social information) 

Launch SocialInfo_ParamEstimate.m
SocialInfo_ParamEstimate.m will read a sample data file, bhv_data.mat.
bhv_data.mat includes sample judgments of a subject for 24 trials.
Data coding infomation is shown in SocialInfo_ParamEstimate.m
SocialInfo_ParamEstimate.m will call SocialInfo_Bayesian.m.
The best parameter will be selected from m_sample x n_sample dimenions parameter space (default 100 x 5 = 500 times) while varying the initial starting point of estimation.
