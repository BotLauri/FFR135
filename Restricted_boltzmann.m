clear

% Initialization of the variables. 
N = 3;
M = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20];
%M = 9;
eta = 0.005;
k = 100;
D_KL_limit = zeros(1, length(M));
p_B_sum = zeros(1, 8);
D_KL_sum = zeros(1, length(M));

% Create the XOR patterns. 
% P(x) = 1/4 for x, 1 to 4. 
% P(x) = 0 for x, 5 to 8.
x_1 = [-1, -1, -1];
x_2 = [1, -1, 1];
x_3 = [-1, 1, 1];
x_4 = [1, 1, -1];
x_5 = [1, 1, 1];
x_6 = [-1, -1, 1];
x_7 = [1, -1, -1];
x_8 = [-1, 1, -1];
x = [x_1; x_2; x_3; x_4; x_5; x_6; x_7; x_8];
p_data = [1/4, 1/4, 1/4, 1/4, 0, 0, 0, 0];

tic
for programTrials = 1:500
for m = 1:length(M)
    
    v = zeros(N, 1);
    h = zeros(M(m), 1);
    w = zeros(N, M(m));
    for i = 1:N
        for j = 1:M(m)
            w(i,j) = randn();
        end
    end
    t_v = zeros(N, 1);
    t_h = zeros(M(m), 1);
    
    trials = 2*10^2;
    for T = 1:trials
        dw = zeros(N, M(m));
        dt_v = zeros(N, 1);
        dt_h = zeros(M(m), 1);
        
        for minibatch = 1:20
            index = randi(4); % Probability 1/4 for pattern 1 to 4, 0 otherwise.
            v = x(index, :)';
            v_0 = v;
            b_h0 = (v_0' * w)' - t_h;
            
            % Update the hidden neurons.
            b_h = (v' * w)' - t_h; 
            for i = 1:M(m) 
                r = rand();
                if r < p(b_h(i))
                    h(i) = 1;
                else
                    h(i) = -1;
                end
            end
            
            for t = 1:k % CD-k loop.
                % Update the visible neurons.
                b_v = h' * w' - t_v';
                for j = 1:N
                    r = rand();
                    if r < p(b_v(j))
                        v(j) = 1;
                    else
                        v(j) = -1;
                    end
                end
                
                % Update the hidden neurons.
                b_h = (v' * w)' - t_h;
                for i = 1:M(m)
                    r = rand();
                    if r < p(b_h(i))
                        h(i) = 1;
                    else
                        h(i) = -1;
                    end
                end           
            end
            
            % Compute the deltas. 
            dw = dw + eta*(v_0*tanh(b_h0)' - v*tanh(b_h)');
            dt_v = dt_v - eta*(v_0 - v);
            dt_h = dt_h - eta*(tanh(b_h0) - tanh(b_h));
        end
        
        % Update the weights and threshholds.
        w = w + dw;
        t_v = t_v + dt_v;
        t_h = t_h + dt_h;
    end
    
    % Compute the Kullback-Leibler divergence.
    N_outer = 10^3;
    N_inner = 10^2;
    p_B = zeros(1, 8);
    
    for I = 1:N_outer
        index = randi(8);
        v = x(index, :)';
        
        % Update the hidden neurons.
        b_h = (v' * w)' - t_h;
        for i = 1:M(m) 
            r = rand();
            if r < p(b_h(i))
                h(i) = 1;
            else
                h(i) = -1;
            end
        end
            
        for J = 1:N_inner
            % Update the visible neurons.
            b_v = h' * w' - t_v';
            for j = 1:N
                r = rand();
                if r < p(b_v(j))
                    v(j) = 1;
                else
                    v(j) = -1;
                end
            end
                
            % Update the hidden neurons.
            b_h = (v' * w)' - t_h;
            for i = 1:M(m)
                r = rand();
                if r < p(b_h(i))
                    h(i) = 1;
                else
                    h(i) = -1;
                end
            end
            
            % Check if the vectors are equal.
            for L = 1:8
                if isequal(v', x(L, :))
                    p_B(L) = p_B(L) + 1/(N_outer*N_inner);
                end
            end
        end
    end
    p_B_sum = p_B_sum + p_B;
    temp = 0;
    for i = 1:4
        if (p_B(i) ~= 0)
            temp = temp + p_data(i) * log(p_data(i)/p_B(i));
        end
    end
    D_KL_sum(m) = D_KL_sum(m) + temp;
    D_KL_limit(m) = KL_div(M(m), N);
end
disp(programTrials)
end
toc
disp(p_B_sum / sum(p_B_sum))
hold on
plot(M,D_KL_sum/programTrials, '*-')
plot(M, D_KL_limit)
title('Plot of Kullback-Leibler divergence vs. M')
xlabel('M')
ylabel('D_{KL}')
legend('Experimental D_{KL}', 'Upper bound for D_{KL}')
hold off

% Acceptance probability. 
function prob = p(x)
    prob = 1/(1 + exp(-2*x));
end

% Upper bound for Kullback-Leibler divergence.
function res = KL_div(M, N)
    if (M < (2^(N-1) - 1))
        res = log(2)*(N - floor(log2(M + 1)) - (M + 1)/(2^(floor(log2(M + 1)))));
    else
        res = 0;
    end
end
