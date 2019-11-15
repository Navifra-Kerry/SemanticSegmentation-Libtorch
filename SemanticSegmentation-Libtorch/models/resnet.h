#pragma once
#include <torch/torch.h>
#include <iostream>
#include <sstream>
#include <map>
#include <list>
#include <opencv2/opencv.hpp>

class BottleneckImpl : public torch::nn::Module
{
public:
	std::string module_name = "Bottleneck";
	static constexpr uint32_t expansion{ 4 };

	int64_t stride;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm bn1;
	torch::nn::Conv2d conv2;
	torch::nn::BatchNorm bn2;
	torch::nn::Conv2d conv3;
	torch::nn::BatchNorm bn3;
	torch::nn::Sequential downsample;

	BottleneckImpl(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
		torch::nn::Sequential downsample_ = torch::nn::Sequential(), int64_t previous_dilation = 1);

	BottleneckImpl(int64_t inplanes, int64_t planes, int64_t dilate, int64_t stride_);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Bottleneck);

class ResNetImpl : public torch::nn::Module
{
public:
	enum Architecture { ResNet50, ResNet101 };

	int64_t inplanes = 64;
	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::BatchNorm bn1{ nullptr };

	torch::nn::Functional relu1{ nullptr };
	torch::nn::Functional max_pool1{ nullptr };

	torch::nn::Sequential layer1{ nullptr };
	torch::nn::Sequential layer2{ nullptr };
	torch::nn::Sequential layer3{ nullptr };
	torch::nn::Sequential layer4{ nullptr };

	ResNetImpl();
	ResNetImpl(Architecture architecture = Architecture::ResNet50);

	torch::Tensor  forward(torch::Tensor x);

private:
	int64_t dilate_;
	std::vector<size_t> layers_{ 3, 4, 0, 3 };
	torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1, bool dilate = false);
};

TORCH_MODULE(ResNet);



