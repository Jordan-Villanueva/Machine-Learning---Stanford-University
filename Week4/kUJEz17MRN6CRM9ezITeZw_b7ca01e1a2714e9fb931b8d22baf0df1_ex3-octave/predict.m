function p = predict(Theta1, Theta2, X)
    % Useful values
    m = size(X, 1);
    num_labels = size(Theta2, 1);

    % You need to return the following variables correctly
    p = zeros(size(X, 1), 1);

    % Add ones to the X data matrix
    X = [ones(m, 1) X];

    % Perform forward propagation
    z2 = X * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(m, 1) a2];  % Add bias unit
    z3 = a2 * Theta2';
    h = sigmoid(z3);

    % Get the predicted labels
    [~, p] = max(h, [], 2);
end

