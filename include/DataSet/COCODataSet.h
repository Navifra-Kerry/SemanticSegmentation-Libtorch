#pragma once
#include "COCODataSet.h"
#include "COCODetection.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "../cocoapi/cocoNote.h"

bool has_valid_annotation(std::vector<Annotation> anno);

/*
annFile  = path of annotation file 
root	 = Path to image file ,annotation files have a Only image file names 
cat_list = Category information you want to learn ,  0 is Background, input as numbers
		   Please refer to MS COCO 2017 category more information for the number..
remove_images_without_annotations = Whether to delete an image without comments

Example
auto val_dataset = COCODataSet(data_dir + "annotations\\instances_val2017.json", data_dir + "val2017", true, { 0,17,18 })
	.map(torch::data::transforms::Stack<>());
const size_t va_dataset_size = val_dataset.size().value();

*/
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