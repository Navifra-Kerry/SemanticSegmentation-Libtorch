#include <models/segmentation/SegmentationModel.h>
#include <models/segmentation/DeeplabV3.h>
#include <models/resnet.h>

SegmentationModelImpl::SegmentationModelImpl()
{
	_aux = false;
}

SegmentationModelImpl::~SegmentationModelImpl()
{

}

void SegmentationModelImpl::fcn_resnet50(bool pretrained, int64_t num_classes, bool aux)
{
	_classifier = _make_FCNHead(2048, num_classes);

	if (aux != false)
	{
		_aux = aux;
		_aux_classifier = _make_FCNHead(1024, num_classes);
	}

	ResNet50 Resnet;

	torch::load(Resnet, "resnet50_Python.pt");

	_backbone = IntermediateLayerGetter(IntermediateLayerGetterImpl(std::move(Resnet), { "layer3","layer4" }));

	register_module("backbone", _backbone);
	register_module("classifier", _classifier);
	register_module("aux_classifier", _aux_classifier);
}

void SegmentationModelImpl::fcn_resnet101(bool pretrained, int64_t num_classes, bool aux)
{
	_classifier = _make_FCNHead(2048, num_classes);
	if (aux != false)
	{
		_aux = aux;
		_aux_classifier = _make_FCNHead(1024, num_classes);
	}

	ResNet101 Resnet;

	torch::load(Resnet, "resnet101_Python.pt");

	_backbone = IntermediateLayerGetter(IntermediateLayerGetterImpl(std::move(Resnet), { "layer3","layer4" }));

	register_module("backbone", _backbone);
	register_module("classifier", _classifier);
	register_module("aux_classifier", _aux_classifier);
}

void SegmentationModelImpl::deeplabv3_resnet101(bool pretrained, int64_t num_classes, bool aux)
{
	int64_t in_channels = 2048;
	_classifier = torch::nn::Sequential
	(
		ASPP(ASPPImpl(2048, { 12,24,36 })),
		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(256, 256, 3).padding(1).bias(false)),
		torch::nn::BatchNorm2d(
			torch::nn::BatchNormOptions(256).eps(0.001).momentum(0.01)),
		torch::nn::Functional(torch::relu),
		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(256, num_classes, 1))
	);

	if (aux != false)
	{
		_aux = aux;
		_aux_classifier = _make_FCNHead(1024, num_classes);
	}

	ResNet101 Resnet;	
	torch::load(Resnet, "resnet101_Python.pt");

	_backbone = IntermediateLayerGetter(IntermediateLayerGetterImpl(std::move(Resnet), {"layer3","layer4"}));

	register_module("backbone", _backbone);
	register_module("classifier", _classifier);
	register_module("aux_classifier", _aux_classifier);
}


void SegmentationModelImpl::deeplabv3_resnet50(bool pretrained, int64_t num_classes, bool aux)
{
	int64_t in_channels = 2048;
	_classifier = torch::nn::Sequential
	(
		ASPP(ASPPImpl(2048, { 12,24,36 })),
		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(256, 256, 3).padding(1).bias(false)),
		torch::nn::BatchNorm2d(
			torch::nn::BatchNormOptions(256).eps(0.001).momentum(0.01)),
		torch::nn::Functional(torch::relu),
		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(256, num_classes, 1))
	);

	if (aux != false)
	{
		_aux = aux;
		_aux_classifier = _make_FCNHead(1024, num_classes);
	}

	ResNet50 Resnet;

	torch::load(Resnet, "resnet50_Python.pt");

	_backbone = IntermediateLayerGetter(IntermediateLayerGetterImpl(std::move(Resnet), { "layer3","layer4" }));

	register_module("backbone", _backbone);
	register_module("classifier", _classifier);
	register_module("aux_classifier", _aux_classifier);
}

torch::nn::Sequential SegmentationModelImpl::_make_FCNHead(int64_t in_channels, int64_t channels)
{
	int64_t inter_channels_ = in_channels / 4;

	return 	torch::nn::Sequential(
		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(in_channels, inter_channels_, 3).padding(1).bias(false)),
		torch::nn::BatchNorm2d(
			torch::nn::BatchNormOptions(inter_channels_).eps(0.001).momentum(0.01)),
		torch::nn::Functional(torch::relu),
		torch::nn::Functional(torch::dropout, 0.1, true),
		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(inter_channels_, channels, 1)));
}

std::unordered_map<std::string, torch::Tensor> SegmentationModelImpl::forward(torch::Tensor x)
{
	std::unordered_map<std::string, torch::Tensor> result;

	int64_t h = x.size(2), w = x.size(3);

	auto feature = _backbone->forward(x);

	x = feature[1];

	x = _classifier->forward(x);
	x = torch::upsample_bilinear2d(x, { h,w }, false);
	result.insert(std::make_pair("out", x));

	if (_aux == true)
	{
		x = feature[0];
		x = _aux_classifier->forward(x);
		x = torch::upsample_bilinear2d(x, { h,w }, false);
		result.insert(std::make_pair("aux", x));
	}

	return result;
}