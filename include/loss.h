#pragma once
#include <torch/torch.h>

torch::Tensor criterion(
	std::unordered_map<std::string, torch::Tensor> inputs, torch::Tensor target)
{
	std::map<std::string, torch::Tensor> losses;

	for (auto loss : inputs)
	{
		losses[loss.first] = torch::nll_loss2d(torch::log_softmax(loss.second, 1), target, {}, 1, 255);
	}

	if (losses.size() == 1)
	{
		return losses["out"];
	}

	return losses["out"] + 0.5 * losses["aux"];
}
