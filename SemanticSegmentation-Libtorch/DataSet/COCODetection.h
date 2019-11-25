#pragma once
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/data/example.h>
#include <torch/data/datasets/base.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "../cocoapi/cocoNote.h"

//reference https://github.com/lsrock1/maskrcnn_benchmark.cpp.git

namespace rcnn {
namespace data 
{

class COCODetection : public torch::data::datasets::Dataset<COCODetection, torch::data::Example<cv::Mat, std::vector<Annotation>>> 
{

public:
	COCODetection(std::string root, std::string annFile/*TODO transform=*/);
	torch::data::Example<cv::Mat, std::vector<Annotation>> get(size_t index) override;
	torch::optional<size_t> size() const override;

	std::string _root;
	COCONote _coco;
	std::vector<int> _ids;

	friend std::ostream& operator << (std::ostream& os, const COCODetection& bml);
};

}//data
}//rcnn