function activation_unit_highest(name_layer,idx_unit)
val_dir = '/home/dmasipr/Data/TraitsCVPR2017/Minitest_jpg';
run('/home/cventuraroy/matconvnet-1.0-beta23/matlab/vl_setupnn.m');
%filename = 'big_1.jpg';
%name_layer = 'conv5_3';
%idx_unit = 1;
K_top_videos = 20;

gpuDevice(2);

net = load('../train/data/exp_regression_avgmax_l28_faces/net-epoch-10.mat');
net = net.net;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ;
net.removeLayer('loss');
net.move('gpu') ;
imdb = load('../utils/imdb_trainval_faces.mat');
inputVar = 'x0';

%%eval process

listing = dir(val_dir);
video_names = [];
for i=1:size(listing,1)
   if strcmp(listing(i).name,'.') || strcmp(listing(i).name,'..')
       continue
   end
   video_names = [video_names; listing(i).name];
end
video_names = cellstr(video_names);

max_activation_values = [];

for ii=1:size(video_names)

img1 = zeros(224,224,3,1);
img1 = single(img1);

cur_videoname = video_names(ii);
filename = [];
sublisting = dir([val_dir '/' cur_videoname{1}]);
for jj=1:size(sublisting,1)
if strcmp(sublisting(jj).name,'.') || strcmp(sublisting(jj).name,'..')
       continue
end
	if strcmp(sublisting(jj).name(end-3:end),'.jpg')
		filename = sublisting(jj).name;
		break
	end
end

if ~isempty(filename)
im_ = imread([val_dir '/' cur_videoname{1} '/' filename]);
im_ = imresize(im_,[224,224]);
im_ = single(im_);
       
im_ = bsxfun(@minus,im_,imresize(imdb.averageImage,[224,224])) ;

img1(:,:,:,1) = im_;

net.eval({inputVar,gpuArray(img1)});

inputs = {inputVar,gpuArray(img1)};
v = net.getVarIndex(inputs(1:2:end)) ;

[net.vars(v).value] = deal(inputs{2:2:end}) ;
inputs = [] ;

for l = net.getLayerExecutionOrder()
  net.layers(l).block.forwardAdvanced(net.layers(l)) ;
end

idx_layer = net.getLayerIndex(name_layer);
activation = net.vars(idx_layer+2).value(:,:,idx_unit);
max_val = max(max(activation));
max_activation_values = [max_activation_values max_val];
else
	max_activation_values = [max_activation_values -1000];
end

end

[top_activation_values, idx_top] = sort(max_activation_values,'descend');

top_videonames = video_names(idx_top(1:K_top_videos));

%% generate uniform receptive field
RFsize = 65; % the average actual size of conv5_3
para.gridScale = [14 14]; % conv5_3 of VGG feature map
para.imageScale = [224 224]; % the input image size
para.RFsize = [RFsize RFsize];  
para.plotPointer = 0; % whether to show the generated RF
maskRF = generateRF(para);

thresholdSegmentation = 0.5; % segmentation threshold

output_dir = ['detectors_faces/' name_layer '/unit_' sprintf('%03d', idx_unit)]
mkdir(output_dir); 

for ii=1:size(top_videonames)

img1 = zeros(224,224,3,1);
img1 = single(img1);

cur_videoname = top_videonames(ii);
sublisting = dir([val_dir '/' cur_videoname{1}]);
filename = [];
for jj=1:size(sublisting,1)
if strcmp(sublisting(jj).name,'.') || strcmp(sublisting(jj).name,'..')
       continue
end
	if strcmp(sublisting(jj).name(end-3:end),'.jpg')
		filename = sublisting(jj).name;
		break
	end
end

im_ = imread([val_dir '/' cur_videoname{1} '/' filename]);
im_ = imresize(im_,[224,224]);
im_ = single(im_);
       
im_ = bsxfun(@minus,im_,imresize(imdb.averageImage,[224,224])) ;

img1(:,:,:,1) = im_;

net.eval({inputVar,gpuArray(img1)});

inputs = {inputVar,gpuArray(img1)};
v = net.getVarIndex(inputs(1:2:end)) ;

[net.vars(v).value] = deal(inputs{2:2:end}) ;
inputs = [] ;

for l = net.getLayerExecutionOrder()
  net.layers(l).block.forwardAdvanced(net.layers(l)) ;
end

idx_layer = net.getLayerIndex(name_layer);
activation = net.vars(idx_layer+2).value(:,:,idx_unit);

curFeature_vectorized = activation(:);
maxValue = max(curFeature_vectorized);
IDX_max = find(curFeature_vectorized>maxValue * thresholdSegmentation);

curMask = squeeze(sum(maskRF(IDX_max,:,:),1));
curMask(curMask>0) = 1;

curImg = imread([val_dir '/' cur_videoname{1} '/' filename]);
curImg = im2double(imresize(curImg,para.imageScale));
curSegmentation = repmat(curMask,[1 1 3]).*curImg+0.2*(1- repmat(curMask,[1 1 3])).*curImg;

imwrite(curSegmentation, [output_dir '/' sprintf('%02d',ii) '_' cur_videoname{1} '.png']);

end
