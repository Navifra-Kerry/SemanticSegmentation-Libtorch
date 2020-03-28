#include "DeeplabV3.h"

ASPPConvImpl::ASPPConvImpl(int64_t in_channels, int64_t out_channels, int64_t dilation)
{

	_conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(dilation).dilation(dilation).bias(false));
	_bn1 = torch::nn::BatchNorm2d(out_channels);
	_relu1 = torch::nn::Functional(torch::relu);

	register_module("0", _conv1);
	register_module("1", _bn1);
	register_module("2", _relu1);
}

ASPPConvImpl::~ASPPConvImpl() {}

void ASPPConvImpl::ASPPConvImpl::reset()
{

}

torch::Tensor ASPPConvImpl::forward(const torch::Tensor& x)
{
	torch::Tensor result;
	result = _conv1->forward(x);
	result = _bn1->forward(result);
	result = _relu1->forward(result);

	return result;
}

ASPPPoolingImpl::ASPPPoolingImpl(int64_t in_channels, int64_t out_channels)
{
	_conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false));
	_bn1 = torch::nn::BatchNorm2d(out_channels);
	_relu1 = torch::nn::Functional(torch::relu);

	register_module("1", _conv1);
	register_module("2", _bn1);
	register_module("3", _relu1);
}

void ASPPPoolingImpl::reset()
{

}

torch::Tensor ASPPPoolingImpl::forward(const torch::Tensor& x)
{
	int64_t h = x.size(2), w = x.size(3);
	auto out = torch::adaptive_avg_pool2d(x, x.sizes());
	out = _conv1->forward(out);
	out = _bn1->forward(out);
	out = _relu1->forward(out);

	return torch::upsample_bilinear2d(out, { h,w }, false);
}


ASPPPoolingImpl::~ASPPPoolingImpl() {}

ConvsImpl::ConvsImpl(int64_t in_channels, int64_t out_channels, std::vector<int64_t>  atrous_rates)
{
	_conv1 =
		torch::nn::Sequential
		(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false)),
			torch::nn::BatchNorm2d(out_channels),
			torch::nn::Functional(torch::relu)
		);

	_conv2 = ASPPConv(in_channels, out_channels, atrous_rates[0]);
	_conv3 = ASPPConv(in_channels, out_channels, atrous_rates[1]);
	_conv4 = ASPPConv(in_channels, out_channels, atrous_rates[2]);
	_avgpool = ASPPPooling(in_channels, out_channels);

	register_module("0", _conv1);
	register_module("1", _conv2);
	register_module("2", _conv3);
	register_module("3", _conv4);
	register_module("4", _avgpool);
}
ConvsImpl::~ConvsImpl() {}

ASPPImpl::ASPPImpl(int64_t in_channels, std::vector<int64_t>  atrous_rates)
{
	int64_t out_channels = 256;

	_convs = Convs(in_channels, out_channels, atrous_rates);
	register_module("convs", _convs);

	_project = torch::nn::Sequential
	(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(5 * out_channels, out_channels, 1).bias(false)),
		torch::nn::BatchNorm2d(out_channels),
		torch::nn::Functional(torch::relu),
		torch::nn::Functional(torch::dropout, 0.5, true)
	);

	register_module("project", _project);
}

torch::Tensor  ASPPImpl::forward(const torch::Tensor& x)
{
	std::vector<torch::Tensor> result;

	result.push_back(_convs->_conv1->forward(x));
	result.push_back(_convs->_conv2->forward(x));
	result.push_back(_convs->_conv3->forward(x));
	result.push_back(_convs->_conv4->forward(x));
	result.push_back(_convs->_avgpool->forward(x));

	auto res = torch::cat(result, 1);

	return _project->forward(res);
}

ASPPImpl::~ASPPImpl() {}

void ASPPImpl::reset()
{

}

