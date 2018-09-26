function [J grad] = CostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% Initialize variables
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Currently X is a 5000x400 matrix, Theta1 is a 50x401 matrix, and Theta2 is a 10x51
%We need to get these to match up in size to calculate H
%We add the bias units
X = [ones(m,1) X]; %Now X is a 5000x401 matrix where the first column is all 1's
%remember that m = 5000, the total number of training examples

%Now forward propagate, starting with X
%Remember, a1 = x, z2 = Theta1*a1, a2=g(z2), z3 = Theta2*a2, and a3 = h = g(z3)
z2 = Theta1 * X'; %50x401 * 401x5000
a2 = sigmoid(z2); %50x5000
a2 = [ones(m,1) a2']; %a2' is 5000x50, we add a column of 1's, a2 is then 5000x51
z3 = Theta2 * a2'; %10x51 * 51*5000
h_theta = sigmoid(z3); %10x5000

%Now we need to change our y to have 0's and 1's, not digits 1-10
y_updated = zeros(num_labels, m); %This will be a 10*5000 matrix, same as h_theta
for i = 1:m,
  y_updated(y(i),i)=1;  %This will set 1 row in each column to have a value of 1
                        %This value of 1 will correspond to the spot (e.g. 1 thru 10)
                        %That the value of y was set to.                 
end

%Now we can calculate the cost function
J = (1/m) * sum(sum((-y_updated) .* log(h_theta) - (1-y_updated) .* log(1-h_theta)));

%Now we account for regularization
%Remember that the first column of matricies Theta1 and Theta 2 are bias terms
%We don't regularize the bias terms, so we exclude them here
Theta1NoBias = Theta1(:,2:size(Theta1,2));
Theta2NoBias = Theta2(:,2:size(Theta2,2));

%Now we can calculate the regularization term for the cost J
RegTerm = (lambda/(2*m)) * (sum(sum(Theta1NoBias.^2)) + sum(sum(Theta2NoBias.^2)));
%The first sum(Theta1NoBias.^2) sums all the columns; we have 1 row full of sums
%The second sum then sums all those columns into singular values

%Now we update our cost function value
J = J + RegTerm;


%Now we begin the back propagation

for t = 1:m,

%Step 1: Set input layer values to the t-th training example x_t.
%Perform a feedforward pass, computing activations (z2, a2, z3, a3) for 2nd & 3rd layers
%Remember, need to hadd a +1 term to ensure vectors of activations for layers
%a1 and a2 also include the bias unit
  a1 = X(t,:); %We already added a bias term to X above.  This is a 1x401 matrix
  a1 = a1'; %Make a1 a 401x1 matrix for computation
  z2 = Theta1 * a1; %Theta1 is 50x401, a1 is 401x1
  a2 = [1;sigmoid(z2)]; %51x1 matrix, since we add bias term
  z3 = Theta2 * a2; %Theta2 is 10x51, a2 is 51x1
  a3 = sigmoid(z3); %10x1 matrix, this is eqivalent to h_theta
  
%Step 2: For each output unit k in the output layer, set delta_k = (a_k - y_k)
%y_k is either 0 or 1, indicating if the current training example belongs to that class
  delta_3 = a3 - y_updated(:,t); %calculated y_updated above to only have 0's or 1's

%Step 3: For the hidden layer l = 2, calculate delta_2:
% delta_2 = Theta_2' * delta_3 .* g'(z2)
  delta_2 = (Theta2' * delta_3) .* [1;sigmoidGradient(z2)]; 
%Theta2' * delta_3 is 51x10 * 10x1, SigmoidGradient only 50x1, so we give it a 1 term (now 51x1)
  
%Step 4:
%Accumulate the gradient from this example using the following formula.
%Remember to skip or remove delta_2(0), as the bias unit doesn't have a delta_2
%This can be done with delta_2 = delta_2(2:end)
  delta_2 = delta_2(2:end); %Now a 50x1 matrix again
  Theta2_grad = Theta2_grad + delta_3 * a2'; %Theta2_grad is 10x51
  Theta1_grad = Theta1_grad + delta_2 * a1'; %Theta1_grad is 50x401

end

%Step 5: Now outside the for loop, obtain the unregularized gradient for the
%neural network cost function by dividing the accumulated gradients by 1/m
Theta2_grad = ((Theta2_grad)/m);
Theta1_grad = ((Theta1_grad)/m);

%Now we have to implement regularization to the gradient
%We add regularization by adding the term (lambda/m) * Theta_ij_(l) for j>=1

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
