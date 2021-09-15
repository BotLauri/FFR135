% Stochastic Hopfield network.
clear

% First we initialize the constants.
numberOfPatterns = 7;
numberOfBits = 200;
trials = 10^2;
updates = 2*10^5;
m_total = 0;

for k = 1:trials
    
    % Then we create the random patterns.
    randomPatterns = zeros(numberOfPatterns, numberOfBits);
    for i = 1:numberOfPatterns
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

    % Store original patterns in network using Hebb's rule.
    weights = zeros(numberOfBits, numberOfBits);
    for i = 1:numberOfBits
        for j = 1:numberOfBits
            temp = 0;
            for my = 1:numberOfPatterns
                temp = temp + randomPatterns(my, i) * randomPatterns(my, j);
            end
            weights(i, j) = 1/numberOfBits * temp;
        end
    end

    % Setting the diagonal to zero.
    for i = 1:numberOfBits
        weights(i, i) = 0;
    end

    % Feed the stored pattern x^(1) to the network.
    choosenPattern = randomPatterns(1, :);
    m = 0;
    for a = 1:updates
        randomIndex = randi(numberOfBits);
        temp = dot(weights(randomIndex, :), choosenPattern);
        prob = 1 / (1 + exp(-2*2*temp));
        r = rand();
        if r < prob
            choosenPattern(randomIndex) = 1;
        else
            choosenPattern(randomIndex) = -1;
        end
        m = m + dot(choosenPattern, randomPatterns(1, :))/numberOfBits;
    end
    m_avg = m/updates;
    m_total = m_total + m_avg;
end

disp(m_total/trials)
