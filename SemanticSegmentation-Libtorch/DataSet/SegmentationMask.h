#pragma once
#include <torch/torch.h>
#include "../cocoapi/mask.h"

namespace rcnn {
namespace structures {

enum Flip {
	FLIP_LEFT_RIGHT,
	FLIP_TOP_BOTTOM
};

torch::Tensor ArrayToTensor(coco::Masks mask);

class Polygons {

public:
	Polygons(std::vector<std::vector<double>> polygons, std::pair<int, int> size, std::string mode);
	Polygons(std::vector<torch::Tensor> polygons, std::pair<int, int> size, std::string mode);
	Polygons(const Polygons& other);
	//Polygons(Polygons&& other) noexcept;
	Polygons& operator=(const Polygons& other) = default;
	//Polygons& operator=(Polygons&& other) noexcept;
	Polygons Transpose(const Flip method);
	Polygons Crop(const std::tuple<int, int, int, int> box);
	Polygons Resize(std::pair<int, int> size);
	torch::Tensor GetMaskTensor();

private:
	std::pair<int, int> size_;
	std::string mode_;
	std::vector<torch::Tensor> polygons_;

	friend std::ostream& operator << (std::ostream& os, const Polygons& bml);

};

class SegmentationMask {

public:
	SegmentationMask(std::vector<std::vector<std::vector<double>>> polygons, std::pair<int, int> size, std::string mode);
	SegmentationMask(std::vector<Polygons> polygons, std::pair<int, int> size, std::string mode);
	SegmentationMask(const SegmentationMask& other);
	SegmentationMask(SegmentationMask&& other) noexcept;
	SegmentationMask& operator=(const SegmentationMask& other) = default;
	SegmentationMask& operator=(SegmentationMask&& other) noexcept;

	SegmentationMask Transpose(const Flip method);
	SegmentationMask Crop(const std::tuple<int, int, int, int> box);
	SegmentationMask Crop(torch::Tensor box);
	SegmentationMask Resize(std::pair<int, int> size);
	SegmentationMask to();
	int Length();
	torch::Tensor GetMaskTensor();

	SegmentationMask operator[](torch::Tensor item);
	SegmentationMask operator[](const int64_t item);

private:
	std::vector<Polygons> polygons_;
	std::pair<int, int> size_;
	std::string mode_;

	friend std::ostream& operator << (std::ostream& os, const SegmentationMask& bml);

};
}
}