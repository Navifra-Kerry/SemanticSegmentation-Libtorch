#include "COCODetection.h"

namespace rcnn {
namespace data {

	COCODetection::COCODetection(std::string root, std::string annFile)
		:_root(root),
		_coco(COCONote(annFile))
	{
		_ids.reserve(_coco._imgs.size());
		for (auto& img : _coco._imgs)
			_ids.push_back(img.first);
	}

	torch::data::Example<cv::Mat, std::vector<Annotation>> COCODetection::get(size_t index) 
	{
		int img_id = _ids.at(index);
		std::vector<int64_t> ann_ids = _coco.GetAnnIds(std::vector<int>{img_id});
		std::vector<Annotation> target = _coco.LoadAnns(ann_ids);
		std::string path(_coco.LoadImgs(std::vector<int>{img_id})[0]._file_name);
		cv::Mat img = cv::imread(_root + "/" + path, cv::IMREAD_COLOR);


		torch::data::Example<cv::Mat, std::vector<Annotation>> value{ img, target };
		return value;
	}

	torch::optional<size_t> COCODetection::size() const
	{
		return _ids.size();
	}

	std::ostream& operator << (std::ostream& os, const COCODetection& bml) 
	{
		os << "Dataset COCODetection\n";
		os << "   Number of datapoints: " << bml.size().value() << "\n";
		os << "   Root Location: " << bml._root << "\n";
		return os;
	}

}//data
}//rcnn