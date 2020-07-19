#pragma once

#include <torch/torch.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <functional>
#include <utility>

class ASPPConvImpl : public torch::nn::Cloneable<ASPPConvImpl>
{
public:
	explicit ASPPConvImpl(int64_t in_channels, int64_t out_channels, int64_t dilation);
	~ASPPConvImpl();

	void reset() override;



	torch::Tensor forward(const torch::Tensor& x);

	torch::nn::Conv2d _conv1{ nullptr };
	torch::nn::BatchNorm2d _bn1{ nullptr };
	torch::nn::Functional _relu1{ nullptr };
};

TORCH_MODULE(ASPPConv);

class ASPPPoolingImpl : public torch::nn::Cloneable<ASPPPoolingImpl>
{
public:
	explicit ASPPPoolingImpl(int64_t in_channels, int64_t out_channels);
	~ASPPPoolingImpl();

	void reset() override;

	torch::Tensor forward(const torch::Tensor& x);

	torch::nn::Conv2d _conv1{ nullptr };
	torch::nn::BatchNorm2d _bn1{ nullptr };
	torch::nn::Functional _relu1{ nullptr };
};

TORCH_MODULE(ASPPPooling);

class ConvsImpl : public torch::nn::Module
{
public:
	ConvsImpl(int64_t in_channels, int64_t out_channels = 256, std::vector<int64_t>  atrous_rates = { 12,24,36 });
	~ConvsImpl();

	torch::nn::Sequential _conv1{ nullptr };
	ASPPConv _conv2{ nullptr };
	ASPPConv _conv3{ nullptr };
	ASPPConv _conv4{ nullptr };
	ASPPPooling _avgpool{ nullptr };
};

TORCH_MODULE(Convs);

class ASPPImpl : public torch::nn::Cloneable<ASPPImpl>
{
public:
	explicit ASPPImpl(int64_t in_channels, std::vector<int64_t>  atrous_rates = { 12,24,36 });
	~ASPPImpl();

	void reset() override;

	torch::Tensor forward(const torch::Tensor& x);

	Convs _convs{ nullptr };
	torch::nn::Sequential _project{ nullptr };
};

TORCH_MODULE(ASPP);


