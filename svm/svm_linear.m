% SVM Email text classification
clear all ; close all ; clc
% Load training features and labels
[ train_y , train_x ] = libsvmread ( 'email_train-all.txt' );
% Train the model and get the primal variables w, b from the model
% Libsvm options
% -t 0 : linear kernel
% Leave other options as their defaults
% model = svmtrain(train_y, train_x, '-t 0');
% w = model.SVs' * model.sv_coef;
% b = -model.rho;
% if (model.Label(1) == -1)
% w = -w; b = -b;
% end
model = svmtrain ( train_y , train_x , sprintf ( '-s 0 -t 0 -c 1' ));
% Load testing features and labels
[ test_y , test_x ] = libsvmread ( 'email_test.txt' );
[ predicted_label , accuracy , decision_values ] = svmpredict ( test_y , test_x , model );
% After running svmpredict, the accuracy should be printed to the matlab
% console


iris = load('iris.csv');
data = iris(:, 1:4);
target = iris(:, 5);
[X_train, y_train,  X_test, y_test] = split_train_test(data, target, 3, 0.8);
model = svmtrain(y_train , X_train, '-s 0 -t 0 -c 5');
[ predicted_label , accuracy , decision_values ] = svmpredict(y_test, X_test, model);
