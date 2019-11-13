#include "coco.h"
#include "mask.h"
#include <fstream>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <ctime>

using namespace Poco::JSON;

namespace coco{

Annotation::Annotation(const Object* j)
{
	id = j->get("id").convert<int64_t>();
	image_id = j->get("image_id").convert<int>();
	category_id = j->get("category_id").convert<int>();
	area = j->get("area").convert<double>();
	iscrowd = j->get("iscrowd").convert<bool>();
	if(j->get("segmentation").isStruct())
	{
		Object::Ptr a = j->get("segmentation").extract<Object::Ptr>();

		if(a->get("counts").isArray())
		{//uncompressed rle
			for(auto& coord : a->get("counts"))
				counts.push_back(coord.convert<int>());

			Array::Ptr s = a->get("size").extract<Array::Ptr>();
			size = std::make_pair(s->get(0).extract<int>(), s->get(1).extract<int>());
		}
		else if(a->get("counts").isString())
		{//compressed rle
			compressed_rle = a->get("counts").convert<std::string>();
			Array::Ptr s = a->get("size").extract<Array::Ptr>();
			size = std::make_pair(s->get(0).extract<int>(), s->get(1).extract<int>());
		}
		else
		{
			assert(false);
		}
	}
	else if(j->get("segmentation").isArray())
	{
		for(auto& polygon : j->get("segmentation"))
		{
		  std::vector<double> tmp;
		  for(auto& coord : polygon)
			tmp.push_back(coord.convert<double>());
		  segmentation.push_back(tmp);
		}
	}
    else
	{
		assert(false);
	}
  
	for(auto& bbox_value : j->get("bbox"))
	  bbox.push_back(bbox_value.convert<double>());
}

Annotation::Annotation(): id(0), image_id(0), category_id(0), area(0), iscrowd(0){}

Annotation::Annotation(const Annotation& obj) :
	id(obj.id), image_id(obj.image_id), category_id(obj.category_id),area(obj.area), iscrowd(obj.iscrowd)
{
}

Image::Image(const Object* j)
{
	id = j->get("id").convert<int>();
	width = j->get("width").convert<int>();
	height = j->get("height").convert<int>();
	file_name = j->get("file_name").convert<std::string>();
}

Image::Image(): id(0), width(0), height(0), file_name(""){}

Categories::Categories(const Object* j)
{
	id = j->get("id").extract<int>();
	name = j->get("name").extract<std::string>();
	supercategory = j->get("supercategory").extract<std::string>();
}

Categories::Categories(): id(0), name(""), supercategory("")
{
}

COCO::COCO(std::string annotation_file){
  std::cout << "loading annotations into memory...\n";
  time_t start = time(0);
  std::ifstream ifs(annotation_file);
  
  Poco::JSON::Parser parser;
  std::cout << "Done : " << difftime(time(0), start) << "s\n";
  cocodataset = parser.parse(ifs).extract<Poco::JSON::Object::Ptr>();
  //assert(parser.is);
  CreateIndex();
}

COCO::COCO(){};

COCO::COCO(const COCO& other) :anns(other.anns), imgs(other.imgs), cats(other.cats), imgToAnns(other.imgToAnns), catToImgs(other.catToImgs)
{
  //dataset.CopyFrom(other., dataset.GetAllocator());
}

COCO::COCO(COCO&& other) :anns(other.anns), imgs(other.imgs), cats(other.cats), imgToAnns(other.imgToAnns), catToImgs(other.catToImgs)
{
  //dataset.CopyFrom(other.dataset.Move(), dataset.GetAllocator());
}

COCO COCO::operator=(const COCO& other)
{
  if(this != &other){
    anns = other.anns;
    imgs = other.imgs;
    imgToAnns = other.imgToAnns;
    catToImgs = other.catToImgs;
    
	//dataset.CopyFrom(other.dataset, dataset.GetAllocator());
  }
  return *this;
}

COCO COCO::operator=(COCO&& other){
  if(this != &other){
    anns = other.anns;
    imgs = other.imgs;
    imgToAnns = other.imgToAnns;
    catToImgs = other.catToImgs;
    //dataset.CopyFrom(other.dataset.Move(), dataset.GetAllocator());
  }
  return *this;
}

void COCO::CreateIndex()
{
	std::cout << "creating index...\n";
	if(cocodataset->has("annotations"))
	{
		assert(cocodataset->get("annotations").isArray());


		for(auto& ann : cocodataset->get("annotations"))
		{
			Object::Ptr j = ann.extract<Object::Ptr>();

			if(imgToAnns.count(j->get("image_id").convert<int>()))
			{ // if it exists
				imgToAnns[j->get("image_id").convert<int>()].emplace_back(j);
			}
			else
			{
				imgToAnns[j->get("image_id").convert<int>()] = std::vector<Annotation> {Annotation(j)};
			}

			anns[static_cast<int64_t>(j->get("id").convert<double>())] = Annotation(j);
		}
	}

	if(cocodataset->has("images"))
	{
		for(auto& img : cocodataset->get("images"))
		{
			Object::Ptr j = img.extract<Object::Ptr>();
			imgs[j->get("id").convert<int>()] = Image(j);
		}
	}
	
	if(cocodataset->has("categories"))
	{
	  assert(cocodataset->get("categories").isArray());

	  for(auto& cat : cocodataset->get("categories"))
	  {
		  Object::Ptr j = cat.extract<Object::Ptr>();
		  cats[j->get("id").convert<int>()] = Categories(j);
	  }
	}
	
	if(cocodataset->has("categories") && cocodataset->has("annotations"))
	{
		for (auto& ann : cocodataset->get("annotations"))
		{
			Object::Ptr j = ann.extract<Object::Ptr>();
			if (catToImgs.count(j->get("category_id").convert<int>()))
			{
				catToImgs[j->get("category_id").convert<int>()].push_back(j->get("image_id").convert<int>());
			}
			else
			{
				catToImgs.insert(
				{
					j->get("category_id").convert<int>(), std::vector<int> {j->get("image_id").convert<int>()} }
				);
			}
		}
	}//ann and cat
  std::cout << "index created!\n";
}

std::vector<int64_t> COCO::GetAnnIds(const std::vector<int> imgIds, 
                           const std::vector<int> catIds, 
                           const std::vector<float> areaRng, 
                           Crowd iscrowd)
{
	std::vector<int64_t> returnAnns;
	std::vector<Annotation> tmp_anns;

	if(imgIds.size() == 0 && catIds.size() == 0 && areaRng.size() == 0)
	{
		for(auto& ann : cocodataset->get("annotations"))
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
				if(imgToAnns.count(imgId))
				{//if it exists
					tmp_anns.insert(tmp_anns.end(), imgToAnns[imgId].begin(), imgToAnns[imgId].end());
				}
			}
		}
		else
		{
			for(auto& ann : cocodataset->get("annotations"))
			{
				tmp_anns.emplace_back(ann.extract<Object::Ptr>());
			}
		}

		if(catIds.size() != 0)
		{
			for(auto it = tmp_anns.begin(); it != tmp_anns.end(); ++it)
			{
				if(std::find(catIds.begin(), catIds.end(), it->category_id) == catIds.end())
				{
					it = tmp_anns.erase(it);
				}
			}
		}

		if(areaRng.size() != 0)
		{
			for(auto it = tmp_anns.begin(); it != tmp_anns.end(); ++it)
			{
				if(it->area <= areaRng[0] || it->area >= areaRng[1]){
					it = tmp_anns.erase(it);
				}
			}
		}
	}

	if(iscrowd == none)
	{
		for(auto& i : tmp_anns)
		{
			returnAnns.push_back(i.id);
		}
	}
	else
	{
		bool check = (iscrowd == F ? false : true);
		for(auto& i : tmp_anns)
		{
			if(i.iscrowd == check)
			returnAnns.push_back(i.id);
		}
	}
	return returnAnns;
}

std::vector<int> COCO::GetCatIds(const std::vector<std::string> catNms, 
                                 const std::vector<std::string> supNms, 
                                 const std::vector<int> catIds)
{
	std::vector<int> returnIds;
	std::vector<Categories> cats;
	for(auto& cat: cocodataset->get("categories"))
	{
		cats.emplace_back(cat.extract<Object::Ptr>());
	}

	if(catNms.size() != 0)
	{
		for(auto it = cats.begin(); it != cats.end();)
		{
			if (std::find(catNms.begin(), catNms.end(), it->name) == catNms.end())
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
			if (std::find(supNms.begin(), supNms.end(), it->supercategory) == supNms.end())
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
			if (std::find(catIds.begin(), catIds.end(), it->id) == catIds.end())
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
		returnIds.push_back(cat.id);
	}
 
	return returnIds;  
}

std::vector<Annotation> COCO::LoadAnns(std::vector<int64_t> ids)
{
	std::vector<Annotation> returnAnns;

	for (auto& id : ids)
	{
		returnAnns.push_back(anns[id]);
	}

	return returnAnns;
}

std::vector<Image> COCO::LoadImgs(std::vector<int> ids)
{
	std::vector<Image> returnImgs;

	for (auto& id : ids)
	{
		returnImgs.push_back(imgs[id]);
	}

	return returnImgs;
}

//COCO COCO::LoadRes(std::string res_file)
//{
//  COCO res = COCO();
//
//  //it only supports json file
//  std::ifstream ifs(res_file);
//
//  Poco::JSON::Parser parser;
//  cocodataset = parser.parse(ifs).extract<Poco::JSON::Object::Ptr>();
//
//  // std::vector<int> annsImgIds;
//  // for(auto& ann : anno.GetArray())
//  //   annsImgIds.push_back(ann["image_id"].GetInt());
//  //no image id check
//  //no caption implementation
//  if(cocodataset[0].has("bbox") && !cocodataset[0].has("bbox"))
//  {
//    Document::AllocatorType& a = res.dataset.GetAllocator(); 
//    Value copied_categories(dataset["categories"], a);
//    res.dataset.AddMember("categories", copied_categories.Move(), a);
//
//    for(int i = 0; i < anno.Size(); ++i)
//	{
//      Value& bb = anno[i]["bbox"];
//      int x1 = bb[0].GetDouble(), x2 = bb[0].GetDouble() + bb[1].GetDouble(), y1 = bb[1].GetDouble(), y2 = bb[1].GetDouble() + bb[3].GetDouble();
//      if(!anno[i].HasMember("segmentation")){
//        Value seg(kArrayType);
//        Value coords(kArrayType);
//        coords.PushBack(x1, a).PushBack(y1, a)
//              .PushBack(x1, a).PushBack(y2, a)
//              .PushBack(x2, a).PushBack(y1, a)
//              .PushBack(x2, a).PushBack(y2, a);
//        
//        anno[i].AddMember("segmentation", seg.PushBack(coords, a).Move(), a);
//      }
//      anno[i].AddMember("area", bb[2].GetDouble() * bb[3].GetDouble(), a);
//      anno[i].AddMember("id", i+1, a);
//      anno[i].AddMember("iscrowd", 0, a);
//    }
//  }
//  else if(anno[0].HasMember("segmentation")){
//    Document::AllocatorType& a = res.dataset.GetAllocator(); 
//    Value copied_categories(dataset["categories"], a);
//    res.dataset.AddMember("categories", copied_categories.Move(), a);
//    for(int i = 0; i < anno.Size(); ++i){
//      std::vector<RLEstr> rlestr;
//      rlestr.emplace_back(
//        std::make_pair(anno[i]["segmentation"]["size"][0].GetInt(), anno[i]["segmentation"]["size"][1].GetInt()),
//        anno[i]["segmentation"]["counts"].GetString()
//      );
//      
//      std::vector<int64_t> seg = coco::area(rlestr);
//      
//      if(!anno[i].HasMember("bbox")){
//        std::vector<double> bbox = coco::toBbox(rlestr);
//        Value bb(kArrayType);
//        for(auto& i : bbox)
//          bb.PushBack(i, a);
//        anno[i].AddMember("bbox", bb, a);
//      }
//
//      anno[i].AddMember("area", seg[0], a);
//      anno[i].AddMember("id", i+1, a);
//      anno[i].AddMember("iscrowd", 0, a);
//    }
//  }
//  //no keypoints
//
//  Document::AllocatorType& a = res.dataset.GetAllocator(); 
//  Value copied_annotations(anno, a);
//  res.dataset.AddMember("annotations", copied_annotations, a);
//  
//  res.CreateIndex();
//  return res;
//}

}//coco namespace
