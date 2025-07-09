
clear
%[Bias_hf, tVecb] = poissonSpikeGen(100, 1, 1);
%load("poisson_vector_3.mat")
%load("post_train_wts_rigth_2.mat")

% Hiperparametros
tau_m = 0.05;
tau_s = 0.01;
t_i = 5;
u_r = 1.5;
th_v = 1;
res_step = 0.0001;

tau_pre = 0.05;%0.006;
tau_post = 0.01;%0.006;

abs_1 = 1e-4;%20e-11;
abs_2 = 1e-4;%20e-11;

m_1 = -abs_1/tau_pre;
m_2 = -abs_1/tau_post;

time_window = 10;
freq = 5;

% Creación del vector de tiempo
vector=[0:res_step:time_window];
T = length(vector);

% Input spikes
Ent_1 = poissonSpikeGenMod(25, time_window, 1);
Ent_2 = poissonSpikeGenMod(25, time_window, 1);
Ent_3 = poissonSpikeGenMod(25, time_window, 1);

z_in_1 = [find(Ent_1==1) numel(Ent_1)]*res_step;
z_in_2 = [find(Ent_2==1) numel(Ent_2)]*res_step;
z_in_3 = [find(Ent_3==1) numel(Ent_3)]*res_step;

s_num_1 = length(z_in_1);
s_num_2 = length(z_in_2);
s_num_3 = length(z_in_3);

weights_d = ([0.3,0.4,0.12]);
des_out = zeros(1,T);
an_sum_d = zeros(1,T);

d_tf = 0; 

% Heaviside function
function H_X= heavisidefunc(x)
    if x<= 0
        H_X=0;
    else
        H_X=1;
    end
end

% Membrane potential behaviour
function [eps] = epsilon_funct(t,t_i,tau_m, tau_s)
if t_i <= t
    eps = ((exp(-max([-t_i,0]/tau_s))/(1-(tau_s/tau_m)))*(exp(-min([t-t_i,t])/tau_m)-exp(-min([t-t_i,t])/tau_s)))*heavisidefunc(t-t_i)*heavisidefunc(t);
else
    eps = 0;
end
end

% Low pass filter
function [fil_signal] = low_pass(z_in, vector, tau)
T = length(vector);
z_sum = length(z_in);
an_signal = zeros(1,T);
for i = 1:z_sum - 1
    for j = 1:T
        an_signal(j) = an_signal(j) + exp((z_in(i)-vector(j))/tau)*heavisidefunc(vector(j)-z_in(i));
    end
end
fil_signal = an_signal;
end

% Target computation
for j = 1:T
    % Analog response
    % Input 1
    for i = 1:s_num_1 - 1
        an_sum_d(j) = an_sum_d(j) + weights_d(1)*epsilon_funct(vector(j)-d_tf,z_in_1(i)-d_tf,tau_m, tau_s);
    end
    % Input 2
    for i = 1:s_num_2 - 1
        an_sum_d(j) = an_sum_d(j) + weights_d(2)*epsilon_funct(vector(j)-d_tf,z_in_2(i)-d_tf,tau_m, tau_s);
    end
    % Input 3
    for i = 1:s_num_3 - 1
        an_sum_d(j) = an_sum_d(j) + weights_d(3)*epsilon_funct(vector(j)-d_tf,z_in_3(i)-d_tf,tau_m, tau_s);
    end
    % fire criterium
    if an_sum_d(j) >= th_v
        d_tf = vector(j);
        des_out(j) = 1;
    end
end

target = des_out;
z_des = [find(target==1) numel(target)]*res_step;
L_S = low_pass(z_des, vector, 0.05);

in_weights = rand(1,3)*0.1;
weights = in_weights;
%weights = weights_d;
%weights = post_train_weights;

weight = zeros(3,T);

out_vector = zeros(1,T);

epochs = 150;
loss_vector = zeros(1,epochs);

% Train loop
for epoch = 1:epochs
    an_sum = zeros(1,T);
    out_vector = zeros(1,T);
    t_f = 0;  
    for j = 1:T
    disp([epoch,j])
    % Analog response
    % Input 1
    for i = 1:s_num_1 - 1
        an_sum(j) = an_sum(j) + weights(1)*epsilon_funct(vector(j)-t_f,z_in_1(i)-t_f,tau_m, tau_s);
    end
    % Input 2
    for i = 1:s_num_2 - 1
        an_sum(j) = an_sum(j) + weights(2)*epsilon_funct(vector(j)-t_f,z_in_2(i)-t_f,tau_m, tau_s);
    end
    % Input 3
    for i = 1:s_num_3 - 1
        an_sum(j) = an_sum(j) + weights(3)*epsilon_funct(vector(j)-t_f,z_in_3(i)-t_f,tau_m, tau_s);
    end
    % fire criterium
    if an_sum(j) >= th_v
        t_f = vector(j);
        out_vector(j) = 1;
    end
    % Algoritm section

    % Output spike process for neuron 1
    for i = 1:s_num_1 - 1
        delta = vector(j) - z_in_1(i);
        % Post spike
        if abs(delta) <= tau_post && delta >= 0
            % Output spike
            if vector(j) == t_f
                weights(1) = weights(1) - (m_2*delta + abs_2);
            end
            % Target spike
            if target(j) == 1
                weights(1) = weights(1) + (m_2*delta + abs_2);
            end
        end
    end
    % Output spike process for neuron 2
    for i = 1:s_num_2 - 1
        delta = vector(j) - z_in_2(i);
        % Post spike
        if abs(delta) <= tau_post && delta >= 0
            % Output spike
            if vector(j) == t_f
                weights(2) = weights(2) - (m_2*delta + abs_2);
            end
            % Target spike
            if target(j) == 1
                weights(2) = weights(2) + (m_2*delta + abs_2);
            end
        end
    end
    % Output spike process for neuron 3
    for i = 1:s_num_3 - 1
        delta = vector(j) - z_in_3(i);
        % Post spike
        if abs(delta) <= tau_post && delta >= 0
            % Output spike
            if vector(j) == t_f
                weights(3) = weights(3) - (m_2*delta + abs_2);
            end
            % Target spike
            if target(j) == 1
                weights(3) = weights(3) + (m_2*delta + abs_2);
            end
        end
    end
    % Bounds for synaptic parameters
    % Conditions for weight 1
    if weights(1) <= 0 
        weights(1) = 0;
    elseif weights(1) >= 1 
        weights(1) = 1;
    end
    % Conditions for weight 2
    if weights(2) <= 0 
        weights(2) = 0;
    elseif weights(2) >= 1 
        weights(2) = 1;
    end
    % Conditions for weight 3
    if weights(3) <= 0 
        weights(3) = 0;
    elseif weights(3) >= 1 
        weights(3) = 1;
    end
    % Loss computation


    weight(1,j) = weights(1);
    weight(2,j) = weights(2);
    weight(3,j) = weights(3);
    end
    % Loss computation
    z_out = [find(out_vector==1) numel(out_vector)]*res_step;
    L_SO = low_pass(z_out, vector, 0.05); % L(S(t))
    loss_vector(epoch) = trapz(vector,abs(L_S-L_SO)); % Integral
end



figure
subplot(4,1,1)
plot ((0:T-1),out_vector);ylabel('Impulsos de entrada d(t_j)');
subplot(4,1,2)
plot ((0:T-1),an_sum);ylabel('Potencial de membrana');
subplot(4,1,3)
plot ((0:T-1),target);ylabel('Impulsos de entrada d(t_j)');
subplot(4,1,4)
plot ((0:T-1),an_sum_d);ylabel('Evolución del peso sinaptico');

figure
subplot(3,1,1)
plot ((0:T-1),weight(1,:));ylabel('Evolución del peso sinaptico 1');
subplot(3,1,2)
plot ((0:T-1),weight(2,:));ylabel('Evolución del peso sinaptico 2');
subplot(3,1,3)
plot ((0:T-1),weight(3,:));ylabel('Evolución del peso sinaptico 3');

figure
plot ((0:T-1),weight);ylabel('Evolución del peso sinaptico');

figure
plot ((0:epochs-1),loss_vector);ylabel('Perdida');

disp(['Desired weights: ',num2str(weights_d(1)),' ',num2str(weights_d(2)),' ',num2str(weights_d(3))])
disp(['Initial weights: ',num2str(in_weights(1)),' ',num2str(in_weights(2)),' ',num2str(in_weights(3))])
disp(['Post train weights: ',num2str(weights(1)),' ',num2str(weights(2)),' ',num2str(weights(3))])

post_train_weights = weights;