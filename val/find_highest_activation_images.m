predicted_values_test = csvread('predictions_reg_avg_max_l28_carles_localization_normalization.csv',1,1);

extraversion_values = predicted_values_test(:,1);
agreeableness_values = predicted_values_test(:,2);
conscientiousness_values = predicted_values_test(:,3);
neuroticism_values = predicted_values_test(:,4);
openness_values = predicted_values_test(:,5);

[extraversion_sorted, idx_extraversion] = sort(extraversion_values,'descend');
[agreeableness_sorted, idx_agreeableness] = sort(agreeableness_values,'descend');
[conscientiousness_sorted, idx_conscientiousness] = sort(conscientiousness_values,'descend');
[neuroticism_sorted, idx_neuroticism] = sort(neuroticism_values,'descend');
[openness_sorted, idx_openness] = sort(openness_values,'descend');

val_dir = '../../test_jpg';
listing = dir(val_dir);
video_names = [];
for i=1:size(listing,1)
   if strcmp(listing(i).name,'.') || strcmp(listing(i).name,'..')
       continue
   end
   video_names = [video_names; listing(i).name];
end
video_names = cellstr(video_names);

%extraversion_highest = video_names(idx_extraversion(1:50));
extraversion_lowest = video_names(idx_extraversion(end:-1:end-49));
%agreeableness_highest = video_names(idx_agreeableness(1:50));
agreeableness_lowest = video_names(idx_agreeableness(end:-1:end-49));
%conscientiousness_highest = video_names(idx_conscientiousness(1:50));
conscientiousness_lowest = video_names(idx_conscientiousness(end:-1:end-49));
%neuroticism_highest = video_names(idx_neuroticism(1:50));
neuroticism_lowest = video_names(idx_neuroticism(end:-1:end-49));
%openness_highest = video_names(idx_openness(1:50));
openness_lowest = video_names(idx_openness(end:-1:end-49));

%save('highest_activation_images.mat','extraversion_highest', 'agreeableness_highest', 'conscientiousness_highest', 'neuroticism_highest', 'openness_highest');

save('lowest_activation_images.mat','extraversion_lowest', 'agreeableness_lowest', 'conscientiousness_lowest', 'neuroticism_lowest', 'openness_lowest');
