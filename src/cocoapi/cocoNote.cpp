#include <cocoapi/cocoNote.h>
#include <cocoapi/mask.h>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <filesystem>

using namespace Poco::JSON;

Annotation::Annotation(const Object* j)
{
	_id = j->get("id").convert<int64_t>();
	_image_id = j->get("image_id").convert<int>();
	_category_id = j->get("category_id").convert<int>();
	_area = j->get("area").convert<double>();
	_iscrowd = j->get("iscrowd").convert<bool>();
	if(!j->get("segmentation").isEmpty() &&  !j->get("segmentation").isArray())
	{
		Object::Ptr o = j->get("segmentation").extract<Object::Ptr>();

		if(o->get("counts").isArray())
		{//uncompressed rle

			Array::Ptr a = o->get("counts").extract<Array::Ptr>();

			for (int i = 0; i < a->size(); i++)
			{
				_counts.push_back(a->get(i).convert<int>());
			}

			Array::Ptr s = o->get("size").extract<Array::Ptr>();
			_size = std::make_pair(s->get(0).convert<int>(), s->get(1).convert<int>());
		}
		else if(o->get("counts").isString())
		{//compressed rle
			_compressed_rle = o->get("counts").convert<std::string>();
			Array::Ptr s = o->get("size").extract<Array::Ptr>();
			_size = std::make_pair(s->get(0).convert<int>(), s->get(1).convert<int>());
		}
		else
		{
			assert(false);
		}
	}
	else if(j->get("segmentation").isArray())
	{
		Array::Ptr a = j->get("segmentation").extract<Array::Ptr>();

		for(int i = 0; i < a->size(); i++)
		{
			Array::Ptr coord = a->get(i).extract<Array::Ptr>();
			std::vector<double> tmp;
			for (int c = 0; c<coord->size(); c++)
			{
				tmp.push_back(coord->get(c).convert<double>());
			}
		  _segmentation.push_back(tmp);
		}
	}
    else
	{
		assert(false);
	}
  
	{
		Array::Ptr a = j->get("bbox").extract<Array::Ptr>();

		for (int i = 0; i < a->size(); i++)
		{
			_bbox.push_back(a->get(i).convert<double>());
		}
	}
}

Annotation::Annotation(): _id(0), _image_id(0), _category_id(0), _area(0), _iscrowd(0){}

Annotation::Annotation(const Annotation& obj) :
	_id(obj._id), _image_id(obj._image_id), _category_id(obj._category_id), _segmentation(obj._segmentation),
	_counts(obj._counts), _compressed_rle(obj._compressed_rle), _size(obj._size), _area(obj._area), _bbox(obj._bbox),
	_iscrowd(obj._iscrowd)
{
}

Image::Image(const Object* j)
{
	_id = j->get("id").convert<int>();
	_width = j->get("width").convert<int>();
	_height = j->get("height").convert<int>();
	_file_name = j->get("file_name").convert<std::string>();
}

Image::Image(): _id(0), _width(0), _height(0), _file_name("") {}

Categories::Categories(const Object* j)
{
	_id = j->get("id").convert<int>();
	_name = j->get("name").convert<std::string>();
	_supercategory = j->get("supercategory").convert<std::string>();
}

Categories::Categories(): _id(0), _name(""), _supercategory("")
{
}

COCONote::COCONote(std::string annotation_file)
{
  std::cout << "loading annotations into memory...\n";
  if (!std::filesystem::exists(annotation_file))
  {
	  std::cout << "The annotation_file does not exist." << std::endl;
	  std::cout << annotation_file << std::endl;
	  quick_exit(1);
  }
  time_t start = time(0);
  std::ifstream ifs(annotation_file);
  
  Poco::JSON::Parser parser;
  _cocodataset = parser.parse(ifs).extract<Poco::JSON::Object::Ptr>();
  //assert(parser.is);
  Parse();
  std::cout << "Done : " << difftime(time(0), start) << "s\n";
}

COCONote::COCONote(){};

COCONote::COCONote(const COCONote& other) 
	:_anns(other._anns), _imgs(other._imgs), _cats(other._cats), _imgToAnns(other._imgToAnns), _catToImgs(other._catToImgs)
{
   _cocodataset = std::move(other._cocodataset);
}

COCONote::COCONote(COCONote&& other) 
	:_anns(other._anns), _imgs(other._imgs), _cats(other._cats), _imgToAnns(other._imgToAnns), _catToImgs(other._catToImgs)
{

	_cocodataset = std::move(other._cocodataset);
}

COCONote COCONote::operator=(const COCONote& other)
{
  if(this != &other){
    _anns = other._anns;
    _imgs = other._imgs;
    _imgToAnns = other._imgToAnns;
    _catToImgs = other._catToImgs;
    
	_cocodataset = std::move(other._cocodataset);
  }
  return *this;
}

COCONote COCONote::operator=(COCONote&& other)
{
  if(this != &other){
    _anns = other._anns;
    _imgs = other._imgs;
    _imgToAnns = other._imgToAnns;
    _catToImgs = other._catToImgs;
	_cocodataset = std::move(other._cocodataset);
  }
  return *this;
}

void COCONote::Parse(std::string annotation_file)
{
	std::cout << "loading annotations into memory...\n";
	time_t start = time(0);
	std::ifstream ifs(annotation_file);

	Poco::JSON::Parser parser;
	_cocodataset = parser.parse(ifs).extract<Poco::JSON::Object::Ptr>();
	//assert(parser.is);
	Parse();
	std::cout << "Done : " << difftime(time(0), start) << "s\n";
}

void COCONote::Parse()
{
#ifdef _DEBUG
	std::cout << "Parse...\n";
#endif
	if(_cocodataset->has("annotations"))
	{
		assert(_cocodataset->get("annotations").isArray());

		Array::Ptr a = _cocodataset->get("annotations").extract<Array::Ptr>();

		for(int i = 0; i < a->size(); i++)
		{
			Object::Ptr j = a->get(i).extract<Object::Ptr>();

			if(_imgToAnns.count(j->get("image_id").convert<int>()))
			{ // if it exists
				_imgToAnns[j->get("image_id").convert<int>()].emplace_back(j);
			}
			else
			{
				_imgToAnns[j->get("image_id").convert<int>()] = std::vector<Annotation> {Annotation(j)};
			}
			
			_anns[static_cast<int64_t>(j->get("id").convert<int64_t>())] = Annotation(j);
		}
	}

	if(_cocodataset->has("images"))
	{
		Array::Ptr a = _cocodataset->get("images").extract<Array::Ptr>();

		for(int i = 0; i < a->size(); i++)
		{
			Object::Ptr j = a->get(i).extract<Object::Ptr>();
			_imgs[j->get("id").convert<int>()] = Image(j);
		}
	}
	
	if(_cocodataset->has("categories"))
	{
	  assert(_cocodataset->get("categories").isArray());

	  Array::Ptr a = _cocodataset->get("categories").extract<Array::Ptr>();

	  for (int i = 0; i < a->size(); i++)
	  {
		  Object::Ptr j = a->get(i).extract<Object::Ptr>();
		  _cats[j->get("id").convert<int>()] = Categories(j);
	  }
	}
	
	if(_cocodataset->has("categories") && _cocodataset->has("annotations"))
	{
		Array::Ptr a = _cocodataset->get("annotations").extract<Array::Ptr>();

		for (int i = 0; i < a->size(); i++)
		{
			Object::Ptr j = a->get(i).extract<Object::Ptr>();

			if (_catToImgs.count(j->get("category_id").convert<int>()))
			{
				_catToImgs[j->get("category_id").convert<int>()].push_back(j->get("image_id").convert<int>());
			}
			else
			{
				_catToImgs.insert(
				{
					j->get("category_id").convert<int>(), std::vector<int> {j->get("image_id").convert<int>()} }
				);
			}
		}
	}//ann and cat

#ifdef _DEBUG
  std::cout << "Parse Complete...\n";
#endif
}

std::vector<int64_t> COCONote::GetAnnIds(const std::vector<int> imgIds,
                           const std::vector<int> catIds, 
                           const std::vector<float> areaRng, 
                           Crowd iscrowd)
{
	std::vector<int64_t> returnAnns;
	std::vector<Annotation> tmp_anns;

	if(imgIds.size() == 0 && catIds.size() == 0 && areaRng.size() == 0)
	{
		for(auto& ann : _cocodataset->get("annotations"))
		{
			tmp_anns.emplace_back(ann.extract<Object::Ptr>());
		}
	}
	else
	{
		if(imgIds.size() != 0)
		{
			for(auto& imgId : imgIds)
			{
				if(_imgToAnns.count(imgId))
				{//if it exists
					tmp_anns.insert(tmp_anns.end(), _imgToAnns[imgId].begin(), _imgToAnns[imgId].end());
				}
			}
		}
		else
		{
			for(auto& ann : _cocodataset->get("annotations"))
			{
				tmp_anns.emplace_back(ann.extract<Object::Ptr>());
			}
		}

		if(catIds.size() != 0)
		{
			for(auto it = tmp_anns.begin(); it != tmp_anns.end(); ++it)
			{
				if(std::find(catIds.begin(), catIds.end(), it->_category_id) == catIds.end())
				{
					it = tmp_anns.erase(it);
				}
			}
		}

		if(areaRng.size() != 0)
		{
			for(auto it = tmp_anns.begin(); it != tmp_anns.end(); ++it)
			{
				if(it->_area <= areaRng[0] || it->_area >= areaRng[1])
				{
					it = tmp_anns.erase(it);
				}
			}
		}
	}

	if(iscrowd == none)
	{
		for(auto& i : tmp_anns)
		{
			returnAnns.push_back(i._id);
		}
	}
	else
	{
		bool check = (iscrowd == F ? false : true);
		for(auto& i : tmp_anns)
		{
			if(i._iscrowd == check)
			returnAnns.push_back(i._id);
		}
	}
	return returnAnns;
}

std::vector<int> COCONote::GetCatIds(const std::vector<std::string> catNms,
                                 const std::vector<std::string> supNms, 
                                 const std::vector<int> catIds)
{
	std::vector<int> returnIds;
	std::vector<Categories> cats;

	Array::Ptr a = _cocodataset->get("categories").extract<Array::Ptr>();

	for (int i = 0; i < a->size(); i++)
	{
		Object::Ptr j = a->get(i).extract<Object::Ptr>();
		cats.emplace_back(j);
	}

	if(catNms.size() != 0)
	{
		for(auto it = cats.begin(); it != cats.end();)
		{
			if (std::find(catNms.begin(), catNms.end(), it->_name) == catNms.end())
			{
				it = cats.erase(it);
			}
			else
			{
				it++;
			}
		}
	}
 
	if(supNms.size() != 0)
	{
		for(auto it = cats.begin(); it != cats.end();)
		{
			if (std::find(supNms.begin(), supNms.end(), it->_supercategory) == supNms.end())
			{
				it = cats.erase(it);
			}
			else
			{
				it++;
			}
		}
	}
 
	if(catIds.size() != 0)
	{
		for(auto it = cats.begin(); it != cats.end();)
		{
			if (std::find(catIds.begin(), catIds.end(), it->_id) == catIds.end())
			{
				it = cats.erase(it);
			}
			else
			{
				it++;
			}
		}
	}

	for(auto& cat: cats)
	{
		returnIds.push_back(cat._id);
	}
 
	return returnIds;  
}

std::vector<Annotation> COCONote::LoadAnns(std::vector<int64_t> ids)
{
	std::vector<Annotation> returnAnns;

	for (auto& id : ids)
	{
		returnAnns.push_back(_anns[id]);
	}

	return returnAnns;
}

std::vector<Image> COCONote::LoadImgs(std::vector<int> ids)
{
	std::vector<Image> returnImgs;

	for (auto& id : ids)
	{
		returnImgs.push_back(_imgs[id]);
	}

	return returnImgs;
}

