% Asynchronous deterministic updates for a Hopfield network.
clear

% First we initialize the constants.
numberOfPatterns = [12, 24, 48, 70, 100, 120];
numberOfBits = 120;
trials = 10^5;
errorP = zeros(1, length(numberOfPatterns));

for a = 1:length(numberOfPatterns)
    for N = 1:trials
        % Then we create the random patterns.
        randomPatterns = zeros(numberOfPatterns(a), numberOfBits);
        for i = 1:numberOfPatterns(a)
            pattern = zeros(1, numberOfBits);
            for j = 1:numberOfBits
                r = rand();
                if r < 0.5
                    pattern(j) = 1;
                else
                    pattern(j) = -1;
                end
            end
            randomPatterns(i, :) = pattern;
        end
        
        % Decide pattern and neuron here for decreased number of
        % calculations. 
        choosenPattern = randomPatterns(1, :);
        randomIndex = randi(numberOfBits);
        
        % Store original patterns in network using Hebb's rule.
        weights = zeros(1, numberOfBits);
        for j = 1:numberOfBits
            temp = 0;
            for my = 1:numberOfPatterns(a)
                temp = temp + randomPatterns(my, randomIndex) * randomPatterns(my, j);
            end
            weights(:, j) = 1/numberOfBits * temp;
        end

        % Setting the diagonal to zero.
        % Comment this for the other results!
        weights(randomIndex) = 0;
    
        % Feed one of the random patterns.
        oldBit = choosenPattern(randomIndex);
        update = sign(dot(weights, choosenPattern));

        % sgn(0) = 1.
        if update == 0
            update = 1;
        end

        % Add to counter if we get an error.
        if oldBit ~= update
            errorP(a) = errorP(a) + 1;
        end
    end
end

disp(numberOfPatterns)
disp(errorP/trials)
