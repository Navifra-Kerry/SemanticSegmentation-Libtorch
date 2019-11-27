#include "cocoDataSet.h"
#include "../cocoapi/mask.h"
#include "../utills/transforms.h"
#include <random> 

std::random_device rd;
std::mt19937 mersenne(rd());
std::uniform_int_distribution<> die(1, 10);


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
	int i = 0;
	for (int i = 0; i < _cat_list.size(); i++)
	{
		_cat_idx.insert(std::make_pair(_cat_list[i], i));
	}

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
		cats.push_back(_cat_idx[obj._category_id]);
	}

	std::vector<torch::Tensor>  mask_tensors;

	//임시
	int base_size = 480;

	//Polygon To Mask Tensors
	for (int k= 0; k< polys.size(); k++)
	{
		//왜 Polygon이 0이지?
		if (polys[k].size() == 0) continue;
		transforms::polygon::Resize((double)base_size / (double)img.cols, (double)base_size / (double)img.rows, polys[k]);

		//임시
		auto frPoly = coco::frPoly(polys[k], base_size, base_size);

		coco::RLEs Rs(1);

		coco::rleFrString(Rs._R, (char*)frPoly[0].counts.c_str(), std::get<0>(frPoly[0].size), std::get<1>(frPoly[0].size));
		coco::siz h = Rs._R[0].h, w = Rs._R[0].w, n = Rs._n;
		coco::Masks masks = coco::Masks(base_size, base_size, 1);

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
	std::tie(target, _) = torch::max(mask_tensor, 0);
	

	cv::resize(img, img, cv::Size(base_size, base_size));


	if (die(mersenne) % 2 == 0)
	{
		target = target.flip({ 1 });
		cv::flip(img, img, 1);
	}

	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 });

#if 0 // Debug Data Inputs
	std::cout << img_tensor.sizes() << std::endl;
	std::cout << target.sizes() << std::endl;

	cv::Mat bin_mask = cv::Mat::eye(target.size(0), target.size(1), CV_8UC1);
	target = target.clamp(0, 255).to(torch::kU8);
	target = target.to(torch::kCPU);
	std::memcpy(bin_mask.data, target.data_ptr(), sizeof(torch::kU8) * target.numel());

	uchar* data_ptr = (uchar*)bin_mask.data;

	for (int y = 0; y < bin_mask.rows; y++)
	{
		for (int x = 0; x < bin_mask.cols; x++)
		{
			if (data_ptr[y * bin_mask.cols + x] == 0)
			{
				continue;
			}
			else
			{
				data_ptr[y * bin_mask.cols + x]  = 255;
			}
		}
	}

	cv::imshow("Image", bin_mask);
	cv::imshow("Image2", img);
	cv::waitKey(0);
#endif

	return { img_tensor.clone(), target.clone() };
}


torch::optional<size_t>  COCODataSet::size() const
{
	return _coco_detection.size();
}

