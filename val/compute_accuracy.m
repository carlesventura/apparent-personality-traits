predicted_values_test = csvread('predictions_reg_avg_max_l28_carles_localization_normalization_faces.csv',1,1);
ground_truth_values_test = load('../utils/ground_truth_test_values.mat');
ground_truth_values_test = ground_truth_values_test.M;
diff_abs = abs(ground_truth_values_test - predicted_values_test);
accuracy = 1 - diff_abs;
mean_accuracy = mean(accuracy,1);
