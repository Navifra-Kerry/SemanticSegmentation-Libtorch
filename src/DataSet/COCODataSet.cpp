#include <DataSet/COCODataSet.h>
#include <cocoapi/mask.h>
#include <utills/transforms.h>
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
	:_coco_detection(root, annFile), _cat_list(cat_list), normalizeChannels({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 })
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
	
	//Get Annotation informationand delete the information except the category I want to learn.
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

	// Annotation information is in Polygon type data type.
	// Loop that converts type into matrix structure of H* W type.
	// In the Matrix information, the corresponding category area has a value of 1, The rest are filled with values ​​of zero
	// The polygon information in the COCO DataSet x1,y1,x2,y2,x3,y3,xn,yn 	Type is of type Double Array..
	std::vector<int> cats;
	std::vector<std::vector<std::vector<double>>> polys;
	for (auto& obj : anno)
	{
		polys.push_back(obj._segmentation);
		cats.push_back(_cat_idx[obj._category_id]);
	}

	std::vector<torch::Tensor>  mask_tensors;

	//The input size of the image and mask is the Base Size for changing to 480..
	int base_size = 480;

	//Polygon To Mask Tensors
	for (int k= 0; k< polys.size(); k++)
	{
		// continue if the size of Polygon is 0
		if (polys[k].size() == 0) continue;

		// Resize Polygon after comparing scale between current loaded image and Base.
		transforms::polygon::Resize((double)base_size / (double)img.cols, (double)base_size / (double)img.rows, polys[k]);

		// Convert Polygon information to data type used by coco API to use coco API
		// Return mask information.
		auto frPoly = coco::frPoly(polys[k], base_size, base_size);

		coco::RLEs Rs(1);

		coco::rleFrString(Rs._R, (char*)frPoly[0].counts.c_str(), std::get<0>(frPoly[0].size), std::get<1>(frPoly[0].size));
		coco::siz h = Rs._R[0].h, w = Rs._R[0].w, n = Rs._n;
		coco::Masks masks = coco::Masks(base_size, base_size, 1);

		coco::rleDecode(Rs._R, masks._mask, n);

		int shape = h * w * n;
		torch::Tensor mask_tensor = torch::empty({ shape });

		// Mask is 1 for category, 0 if not, so multiply by category ID to 2
		// non-area is filled with zeros
		float* data1 = mask_tensor.data_ptr<float>();
		for (size_t i = 0; i < shape; ++i) {
			data1[i] = static_cast<float>(masks._mask[i] * cats[k]);
		}

		// After mapping Mask_tensor to Category, change it to the same matrix form between images
		// h * w
		mask_tensor = mask_tensor.reshape({ static_cast<int64_t>(n),static_cast<int64_t>(w),
			static_cast<int64_t>(h) }).permute({ 2, 1, 0 }).squeeze(2);//fortran order h, w, n

		mask_tensors.push_back(mask_tensor);
	}
	
	// mask_tensors changes the Vector to a Tensor Type.
	// converted to n * h * w
	auto mask_tensor = torch::stack(mask_tensors);

	// will merge the Tensors of the form n * h * w into one
	// example 4 * h * w-> h * w;
	// Get only the Max value, not the value of the same area.
	// i.e. combine different category masks into one.
	torch::Tensor target, _;
	std::tie(target, _) = torch::max(mask_tensor, 0);	

	// Resizng the currently loaded image.
	cv::resize(img, img, cv::Size(base_size, base_size));

	// If the random value is a multiple of 2, Horizental Flip is performed on the image and the target.
	if (die(mersenne) % 2 == 0)
	{
		target = target.flip({ 1 });
		cv::flip(img, img, 1);
	}

	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 });

	img_tensor = normalizeChannels(img_tensor);

	// The code is Debug code to check whether Tensor and Tensor are properly input.
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

	// Return image and Tensor in Tuple.
	return { img_tensor.clone(), target.clone() };
}


torch::optional<size_t>  COCODataSet::size() const
{
	return _coco_detection.size();
}

