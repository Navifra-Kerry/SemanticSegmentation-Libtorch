#pragma once

#include <torch/torch.h>
#include "../IntermediateLayerGetter.h"
#include "../resnet.h"

class SegmentationModelImpl :public torch::nn::Module
{
public:
	SegmentationModelImpl();
	~SegmentationModelImpl();

public:
	void fcn_resnet101(bool pretrained = false, int64_t num_classes = 21, bool aux = true);
	void fcn_resnet50(bool pretrained = false, int64_t num_classes = 21, bool aux = true);

	void deeplabv3_resnet101(bool pretrained = false, int64_t num_classes = 21, bool aux = true);
	void deeplabv3_resnet50(bool pretrained = false, int64_t num_classes = 21, bool aux = true);

	std::unordered_map<std::string, torch::Tensor> forward(torch::Tensor x);
	IntermediateLayerGetter backbone_{ nullptr };


	torch::nn::Sequential classifier_{ nullptr };
	torch::nn::Sequential aux_classifier_{ nullptr };
	torch::nn::Sequential _make_FCNHead(int64_t in_channels, int64_t channels);
	bool aux_;
};

TORCH_MODULE(SegmentationModel);