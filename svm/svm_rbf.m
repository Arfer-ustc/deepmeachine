clear all; close all; clc
% Load training features and labels[y, x] = libsvmread('ex8a.txt');
gamma = 0.21;
[ y , x ] = libsvmread ( 'ex8a.txt' );

% Libsvm options
% -s 0 : classification
% -t 2 : RBF kernel
% -g : gamma in the RBF kernel
model = svmtrain(y, x, sprintf('-s 0 -t 2 -g %g', gamma));
% Display training accuracy[predicted_label, accuracy, decision_values] = svmpredict(y, x, model);
% Plot training data and decision boundary
[ test_y , test_x ] = libsvmread ( 'ex8b.txt' );
[ predicted_label , accuracy , decision_values ] = svmpredict ( test_y , test_x , model );

plotboundary(y, x, model);
title(sprintf('\\gamma = %g', gamma), 'FontSize', 14);


iris = load('iris.csv');
data = iris(:, 1:4);
target = iris(:, 5);
[X_train, y_train,  X_test, y_test] = split_train_test(data, target, 3, 0.8);
model = svmtrain(y_train , X_train, '-s 0 -t 2 -c 1 -g 0.5');
[ predicted_label , accuracy , decision_values ] = svmpredict(y_test, X_test, model);



