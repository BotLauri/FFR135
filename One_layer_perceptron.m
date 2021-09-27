clear

% Import the data, inputs in col 1 and 2, target in 3.
trainingSet = readmatrix('training_set.csv');
validationSet = readmatrix('validation_set.csv');

% Normalizing the input by centering (μ = 0) and normalizing their variances to 1 (σ = 1).
trainInputOneNorm = normalize(trainingSet(:, 1));
trainInputTwoNorm = normalize(trainingSet(:, 2));
valInputOneNorm = normalize(validationSet(:, 1));
valInputTwoNorm = normalize(validationSet(:, 2));

% Define the constants.
M1 = 10;
xTraining = [trainInputOneNorm, trainInputTwoNorm];
xValidation = [valInputOneNorm, valInputTwoNorm];
tTraining = trainingSet(:, 3);
tValidation = validationSet(:, 3);
w1 = randn([M1, width(xTraining)]); % Random init ~ N(0,1).
t1 = zeros(M1, 1); % Init to zero.
v1 = zeros(M1, 1);
w2 = randn([M1, 1]);
t2 = 0;
v2 = 0;
learningRate = 0.005;
C = 1;

v_max = 10^8;
for v = 1:v_max
    
    % First randomly choose mu.
    mu = randi(length(xTraining));
    v1 = xTraining(mu, :);
    
	% Propagate forward.
	for j = 1:M1
        v2(j) = tanh(sum(xTraining(mu, :) .* w1(j, :)) - t1(j));
    end
	O = tanh(sum(w2 .* v2') - t2);
    
    % Compute errors for output layer.
    outputError1 = (tTraining(mu) - O)*(1 - (tanh(dot(w2, v2) - t2)^2));
    gPrime = (1 - (tanh(w1 * v1' - t1).^2));
    for i = 1:M1
        outputError2(i) = outputError1 * w2(i) * gPrime(i);
    end
  
	% Update the weights and threshholds.
	w1 = w1 + learningRate * outputError2' * v1;
	w2 = w2 + (learningRate * outputError1' * v2)';
	t1 = t1 - learningRate * outputError2';
	t2 = t2 - learningRate * outputError1;
    
    % The classification error.
    if rem(v, 1000) == 0
        p_val = length(validationSet);
        temp = 0;
        % For each mu in validation set see if the network guesses right.
        for mu = 1:p_val
            v1 = xValidation(mu, :);
            for j = 1:M1
                v2(j) = tanh(sum(xValidation(mu, :) .* w1(j, :)) - t1(j));
            end
            O = tanh(sum(w2 .* v2') - t2);
            temp = temp + abs(sign(O) - tValidation(mu));
        end
        C = temp/(2*p_val);
        disp(C)
    end
    
    if C < 0.116
        csvwrite('w1.csv', w1);
        csvwrite('w2.csv', w2);
        csvwrite('t1.csv', t1);
        csvwrite('t2.csv', t2);
        break
    end
end
