#include "cocoDataSet.h"
#include "../cocoapi/mask.h"

bool has_valid_annotation(std::vector<Annotation> anno) 
{
	if (anno.size() == 0)
		return false;

	int64_t sum = 0;
	for (auto ann = anno.begin(); ann != anno.end(); ann++)
		sum += ann->_area;

	return sum > 1000;
}

COCODataSet::COCODataSet(std::string annFile, std::string root, bool remove_images_without_annotations,
	std::vector<int> cat_list)
	:_coco_detection(root, annFile), _cat_list(cat_list)
{
	std::sort(_coco_detection._ids.begin(), _coco_detection._ids.end());

	if (remove_images_without_annotations)
	{
		std::vector<int> ids;
		for (auto& i : _coco_detection._ids)
		{
			auto ann_ids = _coco_detection._coco.GetAnnIds(std::vector<int> {i});
			std::vector<Annotation> anno = _coco_detection._coco.LoadAnns(ann_ids);

			for (auto ann = anno.begin(); ann != anno.end();)
			{
				if (std::find(cat_list.begin(), cat_list.end(), ann->_category_id) == cat_list.end())
				{
					anno.erase(ann);
				}
				else
				{
					ann++;
				}
			}

			if (has_valid_annotation(anno))
				ids.push_back(i);
		}
		_coco_detection._ids = ids;
	}
}

std::vector<coco::RLE> _frString(std::vector<coco::RLE>& rleObjs)
{
	size_t n = rleObjs.size();
	std::vector<coco::RLE>  R = std::vector<coco::RLE>(n);
	for (size_t i = 0; i < n; ++i)
		coco::rleFrString(&R[i], (char*)rleObjs[i].cnts, rleObjs[i].h, rleObjs[i].w);

	return R;
}

torch::data::Example<> COCODataSet::get(size_t idx)
{
	auto coco_data = _coco_detection.get(idx);
	cv::Mat img = coco_data.data;

	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 });
	
	//Anatation 가져오기
	std::vector<Annotation> anno = coco_data.target;
	for (auto ann = anno.begin(); ann != anno.end();) 
	{
		if (std::find(_cat_list.begin(), _cat_list.end(), ann->_category_id) == _cat_list.end())
		{
			anno.erase(ann);
		}
		else
		{
			ann++;
		}
	}

	//Mask Polygon 과 카테고리 가져 오기
	std::vector<int> cats;
	std::vector<std::vector<std::vector<double>>> polys;
	for (auto& obj : anno)
	{
		polys.push_back(obj._segmentation);
		cats.push_back(obj._category_id);
	}

	std::vector<torch::Tensor>  mask_tensors;

	//Polygon To Mask Tensors
	for (int k= 0; k< polys.size(); k++)
	{
		auto frPoly = coco::frPoly(polys[k], img.rows, img.cols);

		coco::RLEs Rs(1);

		coco::rleFrString(Rs._R, (char*)frPoly[0].counts.c_str(), std::get<0>(frPoly[0].size), std::get<1>(frPoly[0].size));
		coco::siz h = Rs._R[0].h, w = Rs._R[0].w, n = Rs._n;
		coco::Masks masks = coco::Masks(img.cols, img.rows, 1);

		coco::rleDecode(Rs._R, masks._mask, n);

		int shape = h * w * n;
		torch::Tensor mask_tensor = torch::empty({ shape });

		float* data1 = mask_tensor.data_ptr<float>();
		for (size_t i = 0; i < shape; ++i) {
			data1[i] = static_cast<float>(masks._mask[i] * cats[k]);
		}

		mask_tensor = mask_tensor.reshape({ static_cast<int64_t>(n),static_cast<int64_t>(w),
			static_cast<int64_t>(h) }).permute({ 2, 1, 0 }).squeeze(2);//fortran order h, w, n

		mask_tensors.push_back(mask_tensor);
	}
	
	auto mask_tensor = torch::stack(mask_tensors);

	//Mask에 Category mapping
	torch::Tensor target, _;
	std::tie(target,_)= torch::max(mask_tensor, 0);

	///transform 구현 해야함 임시
	target = target.resize_({ 224 , 224});
	img_tensor = img_tensor.resize_({ 3,224,224 });

	return { img_tensor.clone(), target.clone() };
}


torch::optional<size_t>  COCODataSet::size() const
{
	return _coco_detection.size();
}

