% Asynchronous determistic Hopfield network which recognizes digits.
clear

run('givenPatterns.m');
x1_reshaped = reshape(x1, 1, []);
x2_reshaped = reshape(x2, 1, []);
x3_reshaped = reshape(x3, 1, []);
x4_reshaped = reshape(x4, 1, []);
x5_reshaped = reshape(x5, 1, []);
patterns = [x1_reshaped; x2_reshaped; x3_reshaped; x4_reshaped; x5_reshaped];

% Initialize the constants with the dimensions 16x10.
numberOfPatterns = 5;
numberOfBits = 16*10;

% Store original patterns in network using Hebb's rule.
weights = zeros(numberOfBits, numberOfBits);
for i = 1:numberOfBits
    for j = 1:numberOfBits
        temp = 0;
        for my = 1:numberOfPatterns
            temp = temp + patterns(my, i) * patterns(my, j);
        end
        weights(i, j) = 1/numberOfBits * temp;
    end
end

% Setting the diagonal to zero.
for i = 1:numberOfBits
	weights(i, i) = 0;
end

% Question 1.
inputPattern = [[-1, 1, -1, -1, -1, -1, 1, -1, 1, 1];
                [1, -1, -1, -1, 1, -1, -1, 1, 1, -1];
                [-1, 1, 1, -1, 1, -1, -1, 1, -1, 1];
                [-1, 1, 1, -1, 1, -1, -1, -1, -1, -1];
                [1, 1, 1, 1, -1, 1, -1, -1, -1, 1];
                [-1, 1, 1, 1, -1, -1, -1, 1, -1, 1];
                [1, -1, 1, 1, -1, 1, -1, 1, -1, -1];
                [-1, -1, 1, 1, -1, -1, -1, 1, -1, 1];
                [-1, -1, -1, 1, -1, -1, -1, 1, -1, -1];
                [1, -1, -1, 1, 1, 1, -1, 1, 1, -1];
                [1, 1, 1, 1, 1, 1, -1, -1, 1, -1];
                [1, -1, 1, 1, -1, 1, -1, 1, -1, 1];
                [1, 1, 1, 1, -1, 1, 1, 1, 1, 1];
                [-1, -1, -1, -1, -1, 1, -1, 1, -1, 1];
                [1, -1, 1, -1, 1, 1, 1, -1, -1, 1];
                [-1, 1, 1, 1, -1, -1, -1, -1, 1, 1]];
            
%inputPattern = [[-1, 1, -1, -1, -1, -1, 1, -1, 1, 1], [1, -1, -1, -1, 1, -1, -1, 1, 1, -1], [-1, 1, 1, -1, 1, -1, -1, 1, -1, 1], [-1, 1, 1, -1, 1, -1, -1, -1, -1, -1], [1, 1, 1, 1, -1, 1, -1, -1, -1, 1], [-1, 1, 1, 1, -1, -1, -1, 1, -1, 1], [1, -1, 1, 1, -1, 1, -1, 1, -1, -1], [-1, -1, 1, 1, -1, -1, -1, 1, -1, 1], [-1, -1, -1, 1, -1, -1, -1, 1, -1, -1], [1, -1, -1, 1, 1, 1, -1, 1, 1, -1], [1, 1, 1, 1, 1, 1, -1, -1, 1, -1], [1, -1, 1, 1, -1, 1, -1, 1, -1, 1], [1, 1, 1, 1, -1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, 1, -1, 1, -1, 1], [1, -1, 1, -1, 1, 1, 1, -1, -1, 1], [-1, 1, 1, 1, -1, -1, -1, -1, 1, 1]];
            
% Loop through N times.
states = reshape(inputPattern, 1, []);
updates = 10^3;
for a = 1:updates
    currentStates = states;
    for i = 1:length(states)
        states(i) = sign(dot(weights(i, :), currentStates));
        if states(i) == 0
            states(i) = 1;
        end
    end
end 
states_reshaped = reshape(states, 16, 10);
imshow(states_reshaped)

%% Question 2.
inputPattern2 = [[1, 1, 1, -1, -1, -1, -1, 1, 1, 1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1];
                 [1, 1, 1, -1, -1, -1, -1, 1, 1, 1]];
            
% Loop through N times.
states = reshape(inputPattern2, 1, []);
updates = 10^3;
for a = 1:updates
    currentStates = states;
    for i = 1:length(states)
        states(i) = sign(dot(weights(i, :), currentStates));
        if states(i) == 0
            states(i) = 1;
        end
    end
end 
states_reshaped = reshape(states, 16, 10);
imshow(states_reshaped)

%% Question 3.

inputPattern3 = [[1, 1, 1, -1, -1, 1, -1, 1, 1, -1];
                 [1, 1, 1, -1, -1, 1, -1, 1, 1, -1];
                 [1, 1, 1, -1, -1, 1, -1, 1, 1, -1];
                 [1, 1, 1, -1, -1, 1, -1, 1, 1, -1];
                 [1, 1, 1, -1, -1, 1, -1, 1, 1, -1];
                 [1, 1, 1, -1, -1, 1, -1, 1, 1, -1];
                 [1, 1, 1, -1, -1, 1, -1, 1, 1, -1];
                 [1, 1, 1, 1, 1, -1, 1, 1, 1, -1];
                 [1, 1, 1, 1, 1, -1, 1, 1, 1, -1];
                 [1, -1, -1, -1, -1, 1, -1, 1, 1, -1];
                 [1, -1, -1, -1, -1, 1, -1, 1, 1, -1];
                 [1, -1, -1, -1, -1, 1, -1, 1, 1, -1];
                 [1, -1, -1, -1, -1, 1, -1, 1, 1, -1];
                 [1, -1, -1, -1, -1, 1, -1, 1, 1, -1];
                 [1, -1, -1, -1, -1, 1, -1, 1, 1, -1];
                 [1, -1, -1, -1, -1, 1, -1, 1, 1, -1]];
             
% Loop through N times.
states = reshape(inputPattern3, 1, []);
updates = 10^3;
for a = 1:updates
    for i = 1:length(states)
        states(i) = sign(dot(states, weights(i, :)));
        if states(i) == 0
            states(i) = 1;
        end
    end
end 
states_reshaped = reshape(states, 16, 10);
imshow(states_reshaped)
