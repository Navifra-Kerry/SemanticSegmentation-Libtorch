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
public:
	COCODataSet(std::string annFile, std::string root, bool remove_images_without_annotations);

	torch::data::Example<> get(size_t index) override;
	torch::optional<size_t> size() const override;

	rcnn::data::COCODetection _coco_detection;
};