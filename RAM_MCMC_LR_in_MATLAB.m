clc;
clear all;
close all;


%%
% Synthetic Data Generation
% --------------------------------------------------------------------
% --------------------------------------------------------------------
rng(0); % Set the random seed
sample_size = 1000;
mu = 0;
sigma_e = 2;
theta_true = [4;50;sigma_e];
n_param = length(theta_true);

x = rand(1, sample_size)*10; 
e = mu + sigma_e * randn(1, sample_size);
y = theta_true(1)*x + theta_true(2) +  e;             % a = 40; b = 50; y = a + b*x
data = [x' y'];

%%

n = 10000; %number of samples
%n = input('Enter number of samples: ');
burn_in = 0.50*n; % Burn in = 50 percent

% Define the RAM algorithm parameters
opt_alpha = 0.234;
gamma = 2/3;
q = makedist('Normal');

theta = rand(n_param,1);


PRIOR_MU = zeros(1,n_param);
PRIOR_COV = eye(n_param)*100;
%Prior_mean = logmvnpdf(theta',zeros(1,n_param),eye(n_param)*5);
% a = rand(1);
% b = rand(1);
% sig = rand(1);
%sig = sigma_e;

M = eye(n_param)*10;

S = chol(M, 'lower');

output_chain = zeros(n, n_param);

stats_accepted_values = 0;

for i = 1:n
    % Step R1: Generate a proposal sample
    if mod(i, 100000) == 0
        fprintf('%d iterations done.\n', i);
    end
 
    U = randn(n_param,1);
    theta_proposal = theta + S*U;


%     a_proposal = theta_proposal(1);
%     b_proposal = theta_proposal(2);
%     sig_proposal = theta_proposal(3); 

        
% Step R2: Calculate the acceptance probability

    
    
    % Calculate the log proposal ratio
    log_proposal_ratio = calculate_log_proposal_ratio(theta,theta_proposal);
    
    % Calculate the log likelihood for the proposal parameters
    log_likelihood_proposal = calculate_log_likelihood(theta_proposal,data);
    
    % Calculate the log likelihood for the current parameters
    log_likelihood_current = calculate_log_likelihood(theta,data);
    
    % Calculate the log prior evaluated at proposal parameters
    log_prior_proposal = logmvnpdf(theta_proposal',PRIOR_MU,PRIOR_COV);
    
    % Calculate the log prior evaluated at the current parameters
    log_prior_current = logmvnpdf(theta',PRIOR_MU,PRIOR_COV);
    
    % Calculate the acceptance probability
    % AA = likelihood_proposal + prior_proposal - likelihood_current - prior_current;
    alpha = min(1, exp(log_likelihood_proposal + log_prior_proposal - log_likelihood_current - log_prior_current + log_proposal_ratio));
    
    % Step R3: Accept or reject the proposal
    
    if alpha > rand(1)
        stats_accepted_values = stats_accepted_values + 1;
        output_chain(i, :) = theta_proposal';
        theta = theta_proposal;
    else
        output_chain(i, :) = theta';  

    end
    
    % Step R4: Update the covariance matrix
    
    eta = min(1, n_param * i^(-gamma));
    fac = (eye(n_param) + eta * (alpha - opt_alpha) * (U * U') / norm(U)^2);
    M = S * fac * S' ;
    S = chol(M, 'lower');
    
end

%output.chain = output_chain;
chain_burn_in = output_chain(burn_in:n,:);
acceptance_rate = stats_accepted_values / n;



%% FIGURES

figure;

% Define the number of subplots and their positions
nSubplots = 3;
positions = [1, 2, 3];

for i = 1:nSubplots
    subplot(1, nSubplots, positions(i));
    histogram(chain_burn_in(:,i), 20);
    title(char('a' + i - 1));
    
    % Set the aspect ratio to be equal (square)
    axis square;
    
    hold on; 
    mean_dataset = mean(chain_burn_in(:,i));
    line([mean_dataset, mean_dataset], ylim, 'Color', 'red', 'LineWidth', 2);
    text(mean_dataset, max(ylim), sprintf('Mean = %.2f', mean_dataset), 'Color', 'red', 'VerticalAlignment', 'top');
    hold off;
end



% CHAIN
figure;plot(chain_burn_in);legend('a','b','log_var');


%regression plot

figure;
scatter(data(:,1),data(:,2),'k.');
xlabel('x');
ylabel('y');
hold on
% % 
a1 = mean(chain_burn_in(:,1));
b1 = mean(chain_burn_in(:,2)); 
x_pred = linspace(0,10,500);
y_pred = a1*x_pred + b1;
plot(x_pred,y_pred,'r',linewidth = 2);
 
legend('data','prediction');



%% ADDITIONAL HELP FUNCTIONS


% log_Proposal_Ratio
function log_proposal_ratio = calculate_log_proposal_ratio(theta,theta_proposal)
    log_proposal_ratio = logmvnpdf(theta',theta_proposal',eye(3)*100);
    log_proposal_ratio = log_proposal_ratio - logmvnpdf(theta_proposal',theta',eye(3)*100);
end


% Likelihood
function log_likelihood = calculate_log_likelihood(theta,data)
    x = data(:,1);
    y = data(:,2);
    mu = theta(1) * x + theta(2);
    var = exp(theta(3));
    log_likelihood = log_normal_pdf(y,mu,var);
end

function log_pdf = log_normal_pdf(x, mu, var)
    N = length(x);
    err = x-mu;
    log_pdf = -0.5*(N*log(2*pi)) - 0.5*N*log(var) - 0.5*(err'*err)/var; 
    %log_pdf = -0.5 * log(2 * pi * sigma^2) - 0.5 * ((x - mu).^2) / (sigma^2);
end

function prior = calculate_prior(a, b,M,sig)
    

    prior_mean = [0;0;0];  % Mean vector for parameters a and b
    prior_covariance = M;  % Covariance matrix
    prior = logmvnpdf([a, b,sig], prior_mean', prior_covariance); %Multivariate Normal Distribution
    %prior = prior + log_gamma_pdf(sig,1,1);

end

function [logp] = logmvnpdf(x,mu,Sigma)

[N,D] = size(x);
const = -0.5 * D * log(2*pi);

xc = bsxfun(@minus,x,mu);


log_det_sigma = 2*sum(log(diag(chol(Sigma))));

term1 = -0.5 * sum((xc / Sigma) .* xc, 2);
term2 = const - 0.5 * log_det_sigma;  
logp = term1' + term2;

end





