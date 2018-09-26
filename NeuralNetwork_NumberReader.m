% Neural Network to identify written numbers (0-9)

% Initialize
clear ; close all; clc

% Data Parameters
input_layer_size  = 400;  % We have 20x20 pixel images, resulting in 400 values per input layer
hidden_layer_size = 50;   % Our hidden layer will have 50 units
num_labels = 10;          % 10 labels, from 1 to 10, to identify numbers 0-9
                          % Note that we have mapped "0" to label 10 (since octave indexes from 1, not 0)
                          
% Load our training data; There are 5000 examples here, so X is 5000x400 matrix. y is 5000x1
load('NumberData.mat');

% m is the number of training examples
m = size(X, 1);

% Randomly select 100 data points to display to help visualize the data
DataSelection = randperm(size(X, 1));
DataSelection = DataSelection(1:100);

%Show an example of a single x value (1 row in X matrix)
numberimage = reshape(X(2000, :), 20,20);
imshow(numberimage)

fprintf('Showing visualization of one data point. Press enter to continue.\n');
pause;

%Randomly initialize our Theta matricies
Theta1 = randn(50, 401) .* .0001;
Theta2 = randn(10, 51);

initial_nn_params = [Theta1(:) ; Theta2(:)];

%Perform the feedforward step of the neural network

%Regularization weight lambda
lambda = 1;

%Define the sigmoid function (our activation function):
function g = sigmoid(z)
  g = 1 ./ (1 + exp(-z));
endfunction

%Define the sigmoid gradient function (needed for backprop):
function g = sigmoidGradient(z)
  g = zeros(size(z));
  g = sigmoid(z) .* (1 - sigmoid(z));
endfunction

%Calculate the cost function
J = CostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
                   
%Set options we'll use for neural network
options = optimset('MaxIter', 500);
                   
% Create "short hand" for the cost function to be minimized
costFunction = @(p) CostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
                                
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
fprintf('\n Calculating optimal Theta Values \n')
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                   
%Make predictions using our calculated Theta values
fprintf('\n Making predictions based on calculated Theta values \n')
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);