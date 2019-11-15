#include "DeeplabV3.h"

ASPPConvImpl::ASPPConvImpl(int64_t in_channels, int64_t out_channels, int64_t dilation)
{

	conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(dilation).dilation(dilation).with_bias(false));
	bn1 = torch::nn::BatchNorm(out_channels);
	relu1 = torch::nn::Functional(torch::relu);

	register_module("0", conv1);
	register_module("1", bn1);
	register_module("2", relu1);
}

ASPPConvImpl::~ASPPConvImpl() {}

void ASPPConvImpl::ASPPConvImpl::reset()
{

}

void ASPPConvImpl::pretty_print(std::ostream& stream) const {};

torch::Tensor ASPPConvImpl::forward(const torch::Tensor& x)
{
	torch::Tensor result;
	result = conv1->forward(x);
	result = bn1->forward(result);
	result = relu1->forward(result);

	return result;
}

ASPPPoolingImpl::ASPPPoolingImpl(int64_t in_channels, int64_t out_channels)
{
	conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).with_bias(false));
	bn1 = torch::nn::BatchNorm(out_channels);
	relu1 = torch::nn::Functional(torch::relu);

	register_module("1", conv1);
	register_module("2", bn1);
	register_module("3", relu1);
}

void ASPPPoolingImpl::reset()
{

}

torch::Tensor ASPPPoolingImpl::forward(const torch::Tensor& x)
{
	int64_t h = x.size(2), w = x.size(3);
	auto out = torch::adaptive_avg_pool2d(x, x.sizes());
	out = conv1->forward(out);
	out = bn1->forward(out);
	out = relu1->forward(out);

	return torch::upsample_bilinear2d(out, { h,w }, false);
}

void ASPPPoolingImpl::pretty_print(std::ostream& stream) const {};

ASPPPoolingImpl::~ASPPPoolingImpl() {}

ConvsImpl::ConvsImpl(int64_t in_channels, int64_t out_channels, std::vector<int64_t>  atrous_rates)
{
	conv1 =
		torch::nn::Sequential
		(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).with_bias(false)),
			torch::nn::BatchNorm(out_channels),
			torch::nn::Functional(torch::relu)
		);

	conv2 = ASPPConv(in_channels, out_channels, atrous_rates[0]);
	conv3 = ASPPConv(in_channels, out_channels, atrous_rates[1]);
	conv4 = ASPPConv(in_channels, out_channels, atrous_rates[2]);
	avgpool = ASPPPooling(in_channels, out_channels);

	register_module("0", conv1);
	register_module("1", conv2);
	register_module("2", conv3);
	register_module("3", conv4);
	register_module("4", avgpool);
}
ConvsImpl::~ConvsImpl() {}

ASPPImpl::ASPPImpl(int64_t in_channels, std::vector<int64_t>  atrous_rates)
{
	int64_t out_channels = 256;

	convs = Convs(in_channels, out_channels, atrous_rates);
	register_module("convs", convs);

	project = torch::nn::Sequential
	(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(5 * out_channels, out_channels, 1).with_bias(false)),
		torch::nn::BatchNorm(out_channels),
		torch::nn::Functional(torch::relu),
		torch::nn::Functional(torch::dropout, 0.5, true)
	);

	register_module("project", project);
}

torch::Tensor  ASPPImpl::forward(const torch::Tensor& x)
{
	std::vector<torch::Tensor> result;

	result.push_back(convs->conv1->forward(x));
	result.push_back(convs->conv2->forward(x));
	result.push_back(convs->conv3->forward(x));
	result.push_back(convs->conv4->forward(x));
	result.push_back(convs->avgpool->forward(x));

	auto res = torch::cat(result, 1);

	return project->forward(res);
}

ASPPImpl::~ASPPImpl() {}

void ASPPImpl::reset()
{

}

/// Pretty prints the `BatchNorm` module into the given `stream`.
void ASPPImpl::pretty_print(std::ostream& stream) const
{

}
