#pragma once
#include "COCODataSet.h"
#include "COCODetection.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "../cocoapi/cocoNote.h"

bool has_valid_annotation(std::vector<Annotation> anno);

class COCODataSet : public torch::data::Dataset<COCODataSet>
{
private:

	std::vector<torch::Tensor> states, labels;
	size_t ds_size;
	torch::data::transforms::Normalize<> normalizeChannels;
public:
	COCODataSet(std::string annFile, std::string root, bool remove_images_without_annotations 
		, std::vector<int> cat_list = std::vector<int>{});

	torch::data::Example<> get(size_t index) override;
	torch::optional<size_t> size() const override;

	rcnn::data::COCODetection _coco_detection;
	std::vector<int> _cat_list;
	std::map<int, int> _cat_idx;
};