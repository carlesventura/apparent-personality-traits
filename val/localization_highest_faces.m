val_dir = '/home/dmasipr/Data/TraitsCVPR2017/Minitest_jpg';
run('/home/cventuraroy/matconvnet-1.0-beta23/matlab/vl_setupnn.m');
%filename = 'big_1.jpg';

load('highest_activation_images_faces.mat')
gpuDevice(1);
net = load('../train/data/exp_regression_avgmax_l28_localization_with_normalization_faces/net-epoch-10.mat');
net = net.net;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ;
net.removeLayer('loss');
net.move('gpu') ;
imdb = load('../utils/imdb_trainval_faces.mat');
inputVar = 'x0';

output_dir = 'extraversion_highest_relu_faces';
mkdir(output_dir);
for i=1:size(extraversion_highest)

	videoname = extraversion_highest(i);
	videoname = videoname{1};
	

%%eval process
img1 = zeros(224,224,3,1);
img1 = single(img1);

filename = [];
sublisting = dir([val_dir '/' videoname ]);
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
im_ = imread([val_dir '/' videoname '/' filename]);
im_ = imresize(im_,[224,224]);
im_ = single(im_);
       
im_ = bsxfun(@minus,im_,imresize(imdb.averageImage,[224,224])) ;
       %im_ = gpuArray(im_);

img1(:,:,:,1) = im_;

net.eval({inputVar,gpuArray(img1)});

inputs = {inputVar,gpuArray(img1)};
v = net.getVarIndex(inputs(1:2:end)) ;

[net.vars(v).value] = deal(inputs{2:2:end}) ;
inputs = [] ;

for l = net.getLayerExecutionOrder()
  net.layers(l).block.forwardAdvanced(net.layers(l)) ;
end

l_fc5 = net.getLayerIndex('fc5');
activation_lastconv = net.vars(l_fc5+2).value;

weights_LR = net.getParam('fc6_conv').value;

weights_LR = reshape(weights_LR, [size(weights_LR,3) size(weights_LR,4)]); 

IDX_category = 1;
[curCAMmapAll] = returnCAMmap(activation_lastconv, weights_LR(:,IDX_category));

curCAMmap_crops = squeeze(curCAMmapAll(:,:,1));
%curCAMmapLarge_crops = imresize(curCAMmap_crops,[224 224]);
curCAMmapLarge_crops = imresize(curCAMmap_crops,16);
%curCAMmap_image = mergeTenCrop(curCAMmapLarge_crops);
curCAMmap_image = im2single(curCAMmapLarge_crops);

curHeatMap = map2jpg(curCAMmap_image, [], 'jet');

im_ = imread([val_dir '/' videoname '/' filename]);
im_ = imresize(im_,[224,224]);
im_ = single(im_);

curHeatMap = im2double(im_/255)*0.2+curHeatMap*0.7;
imwrite(curHeatMap,[output_dir '/' videoname '.png']); 
end
end

output_dir = 'agreeableness_highest_relu_faces';
mkdir(output_dir);
for i=1:size(agreeableness_highest)

	videoname = agreeableness_highest(i);
	videoname = videoname{1};
	

%%eval process
img1 = zeros(224,224,3,1);
img1 = single(img1);

filename = [];
sublisting = dir([val_dir '/' videoname ]);
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
im_ = imread([val_dir '/' videoname '/' filename]);
im_ = imresize(im_,[224,224]);
im_ = single(im_);
       
im_ = bsxfun(@minus,im_,imresize(imdb.averageImage,[224,224])) ;
       %im_ = gpuArray(im_);

img1(:,:,:,1) = im_;

net.eval({inputVar,gpuArray(img1)});

inputs = {inputVar,gpuArray(img1)};
v = net.getVarIndex(inputs(1:2:end)) ;

[net.vars(v).value] = deal(inputs{2:2:end}) ;
inputs = [] ;

for l = net.getLayerExecutionOrder()
  net.layers(l).block.forwardAdvanced(net.layers(l)) ;
end

l_fc5 = net.getLayerIndex('fc5');
activation_lastconv = net.vars(l_fc5+2).value;

weights_LR = net.getParam('fc6_conv').value;

weights_LR = reshape(weights_LR, [size(weights_LR,3) size(weights_LR,4)]); 

IDX_category = 2;
[curCAMmapAll] = returnCAMmap(activation_lastconv, weights_LR(:,IDX_category));

curCAMmap_crops = squeeze(curCAMmapAll(:,:,1));
%curCAMmapLarge_crops = imresize(curCAMmap_crops,[224 224]);
curCAMmapLarge_crops = imresize(curCAMmap_crops,16);
%curCAMmap_image = mergeTenCrop(curCAMmapLarge_crops);
curCAMmap_image = im2single(curCAMmapLarge_crops);

curHeatMap = map2jpg(curCAMmap_image, [], 'jet');

im_ = imread([val_dir '/' videoname '/' filename]);
im_ = imresize(im_,[224,224]);
im_ = single(im_);

curHeatMap = im2double(im_/255)*0.2+curHeatMap*0.7;
imwrite(curHeatMap,[output_dir '/' videoname '.png']); 
end
end

output_dir = 'conscientiousness_highest_relu_faces';
mkdir(output_dir);
for i=1:size(conscientiousness_highest)

	videoname = conscientiousness_highest(i);
	videoname = videoname{1};
	

%%eval process
img1 = zeros(224,224,3,1);
img1 = single(img1);

filename = [];
sublisting = dir([val_dir '/' videoname ]);
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
	im_ = imread([val_dir '/' videoname '/' filename]);
	im_ = imresize(im_,[224,224]);
	im_ = single(im_);
       
	im_ = 	bsxfun(@minus,im_,imresize(imdb.averageImage,[224,224])) 	;
       %im_ = gpuArray(im_);

	img1(:,:,:,1) = im_;

	net.eval({inputVar,gpuArray(img1)});

	inputs = {inputVar,gpuArray(img1)};
	v = net.getVarIndex(inputs(1:2:end)) ;

	[net.vars(v).value] = deal(inputs{2:2:end}) ;
	inputs = [] ;

	for l = net.getLayerExecutionOrder()
  		net.layers(l).block.forwardAdvanced(net.layers(l)) ;
	end

	l_fc5 = net.getLayerIndex('fc5');
	activation_lastconv = net.vars(l_fc5+2).value;

	weights_LR = net.getParam('fc6_conv').value;

	weights_LR = reshape(weights_LR, [size(weights_LR,3) 	size(weights_LR,4)]); 

	IDX_category = 3;
	[curCAMmapAll] = returnCAMmap(activation_lastconv, 	weights_LR(:,IDX_category));

	curCAMmap_crops = squeeze(curCAMmapAll(:,:,1));
	%curCAMmapLarge_crops = imresize(curCAMmap_crops,[224 224]);
	curCAMmapLarge_crops = imresize(curCAMmap_crops,16);
	%curCAMmap_image = mergeTenCrop(curCAMmapLarge_crops);
	curCAMmap_image = im2single(curCAMmapLarge_crops);

	curHeatMap = map2jpg(curCAMmap_image, [], 'jet');

	im_ = imread([val_dir '/' videoname '/' filename]);
	im_ = imresize(im_,[224,224]);
	im_ = single(im_);

	curHeatMap = im2double(im_/255)*0.2+curHeatMap*0.7;
	imwrite(curHeatMap,[output_dir '/' videoname '.png']); 
end
end

output_dir = 'neuroticism_highest_relu_faces';
mkdir(output_dir);
for i=1:size(neuroticism_highest)

	videoname = neuroticism_highest(i);
	videoname = videoname{1};
	

%%eval process
img1 = zeros(224,224,3,1);
img1 = single(img1);

filename = [];
sublisting = dir([val_dir '/' videoname ]);
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
im_ = imread([val_dir '/' videoname '/' filename]);
im_ = imresize(im_,[224,224]);
im_ = single(im_);
       
im_ = bsxfun(@minus,im_,imresize(imdb.averageImage,[224,224])) ;
       %im_ = gpuArray(im_);

img1(:,:,:,1) = im_;

net.eval({inputVar,gpuArray(img1)});

inputs = {inputVar,gpuArray(img1)};
v = net.getVarIndex(inputs(1:2:end)) ;

[net.vars(v).value] = deal(inputs{2:2:end}) ;
inputs = [] ;

for l = net.getLayerExecutionOrder()
  net.layers(l).block.forwardAdvanced(net.layers(l)) ;
end

l_fc5 = net.getLayerIndex('fc5');
activation_lastconv = net.vars(l_fc5+2).value;

weights_LR = net.getParam('fc6_conv').value;

weights_LR = reshape(weights_LR, [size(weights_LR,3) size(weights_LR,4)]); 

IDX_category = 4;
[curCAMmapAll] = returnCAMmap(activation_lastconv, weights_LR(:,IDX_category));

curCAMmap_crops = squeeze(curCAMmapAll(:,:,1));
%curCAMmapLarge_crops = imresize(curCAMmap_crops,[224 224]);
curCAMmapLarge_crops = imresize(curCAMmap_crops,16);
%curCAMmap_image = mergeTenCrop(curCAMmapLarge_crops);
curCAMmap_image = im2single(curCAMmapLarge_crops);

curHeatMap = map2jpg(curCAMmap_image, [], 'jet');

im_ = imread([val_dir '/' videoname '/' filename]);
im_ = imresize(im_,[224,224]);
im_ = single(im_);

curHeatMap = im2double(im_/255)*0.2+curHeatMap*0.7;
imwrite(curHeatMap,[output_dir '/' videoname '.png']); 

end
end

output_dir = 'openness_highest_relu_faces';
mkdir(output_dir);
for i=1:size(openness_highest)

	videoname = openness_highest(i);
	videoname = videoname{1};
	

%%eval process
img1 = zeros(224,224,3,1);
img1 = single(img1);

filename = [];
sublisting = dir([val_dir '/' videoname ]);
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
im_ = imread([val_dir '/' videoname '/' filename]);
im_ = imresize(im_,[224,224]);
im_ = single(im_);
       
im_ = bsxfun(@minus,im_,imresize(imdb.averageImage,[224,224])) ;
       %im_ = gpuArray(im_);

img1(:,:,:,1) = im_;

net.eval({inputVar,gpuArray(img1)});

inputs = {inputVar,gpuArray(img1)};
v = net.getVarIndex(inputs(1:2:end)) ;

[net.vars(v).value] = deal(inputs{2:2:end}) ;
inputs = [] ;

for l = net.getLayerExecutionOrder()
  net.layers(l).block.forwardAdvanced(net.layers(l)) ;
end

l_fc5 = net.getLayerIndex('fc5');
activation_lastconv = net.vars(l_fc5+2).value;

weights_LR = net.getParam('fc6_conv').value;

weights_LR = reshape(weights_LR, [size(weights_LR,3) size(weights_LR,4)]); 

IDX_category = 5;
[curCAMmapAll] = returnCAMmap(activation_lastconv, weights_LR(:,IDX_category));

curCAMmap_crops = squeeze(curCAMmapAll(:,:,1));
%curCAMmapLarge_crops = imresize(curCAMmap_crops,[224 224]);
curCAMmapLarge_crops = imresize(curCAMmap_crops,16);
%curCAMmap_image = mergeTenCrop(curCAMmapLarge_crops);
curCAMmap_image = im2single(curCAMmapLarge_crops);

curHeatMap = map2jpg(curCAMmap_image, [], 'jet');

im_ = imread([val_dir '/' videoname '/' filename]);
im_ = imresize(im_,[224,224]);
im_ = single(im_);

curHeatMap = im2double(im_/255)*0.2+curHeatMap*0.7;
imwrite(curHeatMap,[output_dir '/' videoname '.png']); 

end
end
