function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
% LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
% regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y);

% Calculate the cost function without regularization
J = (1 / (2 * m)) * (X * theta - y)' * (X * theta - y);

% Exclude the bias term for regularization
reg_term = (lambda / (2 * m)) * (theta(2:end)' * theta(2:end));  % Assuming theta(1) is the bias term

% Add the regularization term to the cost function
J = J + reg_term;

% Compute the gradient for linear regression
grad = (1 / m) * X' * (X * theta - y);

% Add the regularization term to the gradient except for the bias term
grad(2:end) = grad(2:end) + (lambda / m) * theta(2:end);

end
