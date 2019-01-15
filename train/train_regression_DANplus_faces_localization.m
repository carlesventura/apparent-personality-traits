function [net, info] = train_regression_DANplus(varargin)
opt.model = 'vgg-face';
run('/home/cventuraroy/matconvnet-1.0-beta23/matlab/vl_setupnn.m');
net = load(['/home/cventuraroy/jobcandidateCHALEARN2017/' opt.model '.mat']);
net = dagnn.DagNN.fromSimpleNN(net);
%net = net.net;
inputVar = 'data' ;

%[opts, varargin] = vl_argparse(opts, varargin) ;
opts.imdbPath = [];
opts.networkType = 'dagnn' ;
for name = {'fc8','prob','fc7','relu7','fc6','relu6','pool5'}
      net.removeLayer(name) ;
end

%add convolutional layer before GAP
fc5 = dagnn.Conv('size',[3 3 512 5],'pad',1,'stride',1,'hasBias',true);
net.addLayer('fc5',fc5,{'x30'},{'fc5'},{'fc5_conv','fc5_bias'});

%add ReLU unit
relu6 = dagnn.ReLU();
net.addLayer('relu6',relu6,{'fc5'},{'relu6'});


%add global average pooling
net.addLayer('pool6_avg',dagnn.Pooling('method','avg','poolSize',[14 14],'stride',1,'pad',0),'relu6','fc5_avg');

%add normalization layer
net.addLayer('norm6_avg',L2NORM(),'fc5_avg','fc5_avg_norm');

%add Dropout layer
relu6 = dagnn.DropOut();
net.addLayer('dropout6',relu6,{'fc5_avg_norm'},{'dropout6'});

%add full connected layer
fc6 = dagnn.Conv('size',[1 1 5 5],'pad',0,'stride',1,'hasBias',true);
net.addLayer('fc6',fc6,{'dropout6'},{'fc6'},{'fc6_conv','fc6_bias'});

%add Activation layer
net.addLayer('prob',dagnn.Sigmoid(),{'fc6'},{'prob'});

%add Loss layer
net.addLayer('loss',dagnn.Loss('loss','pdist'),{'prob','label'},{'objective'});

net.params(27).value = single(0.05*randn(3,3,512,5));
net.params(28).value = single(zeros(1,5)');
net.params(29).value = single(0.05*randn(1,1,5,5));
net.params(30).value = single(zeros(1,5)');


%[opts, varargin] = vl_argparse(opts, varargin) ;
opts.imdbPath = [];
opts.networkType = 'dagnn' ;
opts.train = struct() ;
opts.train.expDir = fullfile('data','exp_regression_avgmax_l28_localization_with_normalization_faces');
opts.train.gpus = [1];
opts.train.batchSize = 32;
%opts.train.batchSize = 16;
opts.train.learningRate = logspace(-3,-6,30);
opts.train.numEpochs = 10;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------


%net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end
imdb=load('../utils/imdb_trainval_faces.mat');
%imdb=imdb.imdb_now;
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------

%images = imdb.images.data(:,:,:,batch) ;
images = [];

for i=1:length(batch)
    im_ = imread(['../../Minitrainval_jpg/' imdb.images.name{batch(i)}]);
    
    im_ = imresize(im_,[224 224]);
    im_ = single(im_);
    im_ = bsxfun(@minus,im_,imresize(imdb.averageImage,[224,224])) ;
    images = cat(4,images,im_);
end
labels(1,1,:,:) = imdb.images.class(batch,:)';
if rand > 0.5, images=fliplr(images) ; end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = [];
for i=1:length(batch)
    im_ = imread(['../../Minitrainval_jpg/' imdb.images.name{batch(i)}]);
    
    im_ = imresize(im_,[224 224]);
    im_ = single(im_);
    im_ = bsxfun(@minus,im_,imresize(imdb.averageImage,[224,224])) ;
    images = cat(4,images,im_);
end
labels(1,1,:,:) = imdb.images.class(batch,:)';
if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'x0', images, 'label', labels} ;
% -------------------------------------------------------------------------

