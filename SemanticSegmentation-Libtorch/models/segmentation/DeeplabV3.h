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

	/// Pretty prints the `BatchNorm` module into the given `stream`.
	void pretty_print(std::ostream& stream) const override;

	torch::Tensor forward(const torch::Tensor& x);

	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::BatchNorm bn1{ nullptr };
	torch::nn::Functional relu1{ nullptr };
};

TORCH_MODULE(ASPPConv);

class ASPPPoolingImpl : public torch::nn::Cloneable<ASPPPoolingImpl>
{
public:
	explicit ASPPPoolingImpl(int64_t in_channels, int64_t out_channels);
	~ASPPPoolingImpl();

	void reset() override;

	/// Pretty prints the `BatchNorm` module into the given `stream`.
	void pretty_print(std::ostream& stream) const override;

	torch::Tensor forward(const torch::Tensor& x);

	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::BatchNorm bn1{ nullptr };
	torch::nn::Functional relu1{ nullptr };
};

TORCH_MODULE(ASPPPooling);

class ConvsImpl : public torch::nn::Module
{
public:
	ConvsImpl(int64_t in_channels, int64_t out_channels = 256, std::vector<int64_t>  atrous_rates = { 12,24,36 });
	~ConvsImpl();

	torch::nn::Sequential conv1{ nullptr };
	ASPPConv conv2{ nullptr };
	ASPPConv conv3{ nullptr };
	ASPPConv conv4{ nullptr };
	ASPPPooling avgpool{ nullptr };
};

TORCH_MODULE(Convs);

class ASPPImpl : public torch::nn::Cloneable<ASPPImpl>
{
public:
	explicit ASPPImpl(int64_t in_channels, std::vector<int64_t>  atrous_rates = { 12,24,36 });
	~ASPPImpl();

	void reset() override;

	/// Pretty prints the `BatchNorm` module into the given `stream`.
	void pretty_print(std::ostream& stream) const override;

	torch::Tensor forward(const torch::Tensor& x);

	Convs convs{ nullptr };
	torch::nn::Sequential project{ nullptr };
};

TORCH_MODULE(ASPP);


