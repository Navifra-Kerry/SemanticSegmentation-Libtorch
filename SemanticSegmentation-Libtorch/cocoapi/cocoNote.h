#pragma once
#include <map>
#include <string>
#include <vector>
#include <Poco/JSON/Object.h>
#include <Poco/SharedPtr.h>
#include <Poco/JSON/Parser.h>

enum Crowd{
  none = -1,
  F = 0,
  T = 1
};

struct Annotation
{
  Annotation(const Poco::JSON::Object* j);
  Annotation(const Annotation& obj);
  Annotation();
  int64_t _id;
  int _image_id;
  int _category_id;
  std::vector<std::vector<double>> _segmentation;
  std::vector<int> _counts;
  std::string _compressed_rle;
  std::pair<int, int> _size;
  float _area;
  std::vector<float> _bbox;
  bool _iscrowd;
};

struct Image
{
  Image(const Poco::JSON::Object* j);
  Image();
  ~Image() {};

  int _id;
  int _width;
  int _height;
  std::string _file_name;
};

struct Categories
{
  Categories(const Poco::JSON::Object* j);
  Categories();
  int _id;
  std::string _name;
  std::string _supercategory;
};

/*
Parser of MS COCO Annatation Files
*/
struct COCONote
{
	COCONote(std::string annotation_file);
	COCONote();

	void Parse();
	void Parse(std::string annotation_file);

	std::vector<int64_t> GetAnnIds(const std::vector<int> imgIds = std::vector<int>{}, const std::vector<int> catIds = std::vector<int>{}, const std::vector<float> areaRng = std::vector<float>{}, Crowd iscrowd=none);
	//info
	std::vector<int> GetCatIds(const std::vector<std::string> catNms = std::vector<std::string>{}, const std::vector<std::string> supNms = std::vector<std::string>{}, const std::vector<int> catIds = std::vector<int>{});
	std::vector<Annotation> LoadAnns(std::vector<int64_t> ids);
	std::vector<Image> LoadImgs(std::vector<int> ids);
	COCONote LoadRes(std::string res_file);
	COCONote(const COCONote& other);
	COCONote(COCONote&& other);
	COCONote operator=(const COCONote& other);
	COCONote operator=(COCONote&& other);

	Poco::JSON::Object::Ptr _cocodataset;

	std::map<int64_t, Annotation> _anns;
	std::map<int, Image> _imgs;
	std::map<int, Categories> _cats;
	std::map<int, std::vector<Annotation>> _imgToAnns;
	std::map<int, std::vector<int>> _catToImgs;
};

