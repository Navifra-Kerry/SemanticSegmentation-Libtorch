#pragma once

#include <torch/torch.h>

class IntermediateLayerGetterImpl : public torch::nn::Module
{
public:

	template <typename Net>
	IntermediateLayerGetterImpl(Net  Module, std::vector<std::string> return_layers)
	{
		//this->to(Module->b);

		for (auto children : Module->named_children())
		{
			_module.insert(children.key(), std::move(children.value()));
			register_module(children.key(), _module[children.key()]);
		}

		_return_layers.swap(return_layers);
	}

	~IntermediateLayerGetterImpl();

	std::vector<torch::Tensor>  forward(torch::Tensor x);

private:
	torch::OrderedDict<std::string, std::shared_ptr<Module>> _module;
	std::vector<std::string> _return_layers;
};

TORCH_MODULE(IntermediateLayerGetter);