#include "IntermediateLayerGetter.h"

IntermediateLayerGetterImpl::~IntermediateLayerGetterImpl()
{

}

std::vector<torch::Tensor> IntermediateLayerGetterImpl::forward(torch::Tensor x)
{
	std::vector<torch::Tensor> results;

	x = _module["conv1"]->as<torch::nn::Conv2d>()->forward(x);	
	x = _module["bn1"]->as<torch::nn::BatchNorm>()->forward(x);
	x = _module["relu1"]->as<torch::nn::Functional>()->forward(x);
	x = _module["max_pool1"]->as<torch::nn::Functional>()->forward(x);

	x = _module["layer1"]->as<torch::nn::Sequential>()->forward(x);
	x = _module["layer2"]->as<torch::nn::Sequential>()->forward(x);
	x = _module["layer3"]->as<torch::nn::Sequential>()->forward(x);
	results.push_back(x);
	x = _module["layer4"]->as<torch::nn::Sequential>()->forward(x);
	results.push_back(x);
	
	return results;
}