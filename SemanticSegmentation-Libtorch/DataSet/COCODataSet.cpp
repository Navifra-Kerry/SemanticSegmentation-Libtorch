#include "cocoDataSet.h"
#include "SegmentationMask.h"

bool has_valid_annotation(std::vector<Annotation> anno) 
{
	if (anno.size() == 0)
		return false;

	return true;
}

COCODataSet::COCODataSet(std::string annFile, std::string root, bool remove_images_without_annotations)
	:_coco_detection(root, annFile)
{
	std::sort(_coco_detection._ids.begin(), _coco_detection._ids.end());
	if (remove_images_without_annotations)
	{
		std::vector<int> ids;
		for (auto& i : _coco_detection._ids)
		{
			auto ann_ids = _coco_detection._coco.GetAnnIds(std::vector<int> {i});
			std::vector<Annotation> anno = _coco_detection._coco.LoadAnns(ann_ids);
			if (has_valid_annotation(anno))
				ids.push_back(i);
		}
		_coco_detection._ids = ids;
	}
}

torch::data::Example<> COCODataSet::get(size_t idx)
{
	auto coco_data = _coco_detection.get(idx);
	cv::Mat img = coco_data.data;

	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 });

	std::vector<Annotation> anno = coco_data.target;
	for (auto ann = anno.begin(); ann != anno.end();) {
		if (ann->_iscrowd)
			anno.erase(ann);
		else
			ann++;
	}

	std::vector<std::vector<std::vector<double>>> polys;
	for (auto& obj : anno)
		polys.push_back(obj._segmentation);
	auto mask = new rcnn::structures::SegmentationMask(polys,
		std::make_pair(static_cast<int64_t>(img.cols), static_cast<int64_t>(img.rows)), "poly");

	return { img_tensor.clone(), mask->GetMaskTensor().clone() };
}


torch::optional<size_t>  COCODataSet::size() const
{
	return _coco_detection.size();
}

