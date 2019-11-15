#include "resnet.h"

BottleneckImpl::BottleneckImpl(int64_t inplanes, int64_t planes, int64_t stride_,
	torch::nn::Sequential downsample_, int64_t previous_dilation)
	: conv1(torch::nn::Conv2dOptions(inplanes, planes* (64 / 64.), 1).with_bias(false).stride(1)),
	bn1(planes* (64 / 64.)),
	conv2(torch::nn::Conv2dOptions(planes* (64 / 64.), planes* (64 / 64.), 3).stride(stride_).dilation(previous_dilation).groups(1).padding(previous_dilation).with_bias(false)),
	bn2(planes* (64 / 64.)),
	conv3(torch::nn::Conv2dOptions(planes* (64 / 64.), planes* expansion, 1).stride(1).with_bias(false)),
	bn3(planes* expansion),
	downsample(downsample_)
{
	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("conv2", conv2);
	register_module("bn2", bn2);
	register_module("conv3", conv3);
	register_module("bn3", bn3);
	stride = stride_;
	if (!downsample->is_empty()) {
		register_module("downsample", downsample);
	}
}

BottleneckImpl::BottleneckImpl(int64_t inplanes, int64_t planes, int64_t dilate, int64_t stride_)

	: conv1(torch::nn::Conv2dOptions(inplanes, planes* (64 / 64.), 1).with_bias(false).padding(0)),
	bn1(planes* (64 / 64.)),
	conv2(torch::nn::Conv2dOptions(planes* (64 / 64.), planes* (64 / 64.), 3).dilation(dilate).groups(1).padding(dilate).with_bias(false)),
	bn2(planes* (64 / 64.)),
	conv3(torch::nn::Conv2dOptions(planes* (64 / 64.), planes* expansion, 1).with_bias(false).stride(1)),
	bn3(planes* expansion)

{
	downsample = torch::nn::Sequential();
	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("conv2", conv2);
	register_module("bn2", bn2);
	register_module("conv3", conv3);
	register_module("bn3", bn3);
	stride = stride_;
	if (!downsample->is_empty()) {
		register_module("downsample", downsample);
	}
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x)
{
	at::Tensor residual(x.clone());

	x = conv1->forward(x);
	x = bn1->forward(x);
	x = torch::relu_(x);

	x = conv2->forward(x);
	x = bn2->forward(x);
	x = torch::relu_(x);

	x = conv3->forward(x);
	x = bn3->forward(x);

	if (!downsample->is_empty()) {
		residual = downsample->forward(residual);
	}
	x += residual;
	x = torch::relu_(x);

	return x;
}

ResNetImpl::ResNetImpl() {}

ResNetImpl::ResNetImpl(Architecture architecture)
{
	if (Architecture::ResNet50 == architecture)
	{
		layers_[2] = 6;
	}
	else
	{
		layers_[2] = 23;
	}

	dilate_ = 1;
	conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).with_bias(false));
	bn1 = torch::nn::BatchNorm(64);

	relu1 = torch::nn::Functional(torch::relu);
	max_pool1 = torch::nn::Functional(torch::max_pool2d, 3, 2, 1,1,false);

	layer1 = _make_layer(64, layers_[0]);
	layer2 = _make_layer(128, layers_[1], 2);
	layer3 = _make_layer(256, layers_[2], 2, true);
	layer4 = _make_layer(512, layers_[3], 2, true);

	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("relu1", relu1);
	register_module("max_pool1", max_pool1);
	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);
}

torch::Tensor
ResNetImpl::forward(torch::Tensor x)
{
	x = conv1->forward(x);
	x = bn1->forward(x);
	x = relu1->forward(x);
	x = max_pool1->forward(x);

	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);

	return x;
}

torch::nn::Sequential ResNetImpl::_make_layer(int64_t planes, int64_t blocks, int64_t stride, bool dilate)
{
	int64_t previous_dilation = dilate_;
	if (dilate == true)
	{
		dilate_ *= stride;
		stride = 1;
	}

	torch::nn::Sequential downsample;
	if (stride != 1 or inplanes != planes * BottleneckImpl::expansion) {
		downsample = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes * BottleneckImpl::expansion, 1).stride(stride)),
			torch::nn::BatchNorm(torch::nn::BatchNormOptions(planes * BottleneckImpl::expansion)
			));
	}

	torch::nn::Sequential layers;
	layers->push_back(BottleneckImpl(inplanes, planes, stride, downsample, previous_dilation));
	inplanes = planes * BottleneckImpl::expansion;

	if (dilate == false)
	{
		for (int64_t i = 0; i < blocks - 1; i++) {
			layers->push_back(BottleneckImpl(inplanes, planes));
		}
	}
	else
	{
		for (int64_t i = 0; i < blocks - 1; i++) {
			layers->push_back(BottleneckImpl(inplanes, planes, dilate_, stride));
		}
	}

	return layers;
}
