#ifndef RESNET_H
#define RESNET_H

#include <torch/torch.h>

template <typename Block>
struct ResNetImpl;

namespace _resnetimpl {
	// 3x3 convolution with padding
	torch::nn::Conv2d conv3x3(
		int64_t in,
		int64_t out,
		int64_t stride = 1,
		int64_t groups = 1,
		int64_t dilation = 1);

	// 1x1 convolution
	torch::nn::Conv2d conv1x1(
		int64_t in, 
		int64_t out, 
		int64_t stride = 1);

	struct BasicBlock : torch::nn::Module {
		template <typename Block>
		friend struct ResNetImpl;

		int64_t stride;
		torch::nn::Sequential downsample;

		torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
		torch::nn::BatchNorm2d bn1{ nullptr }, bn2{ nullptr };

		static int expansion;

		BasicBlock(
			int64_t inplanes,
			int64_t planes,
			int64_t stride = 1,
			torch::nn::Sequential downsample = nullptr,
			int64_t groups = 1,
			int64_t base_width = 64,
			int diation = 1);

		torch::Tensor forward(torch::Tensor x);
	};

	struct Bottleneck : torch::nn::Module {
		template <typename Block>
		friend struct ResNetImpl;

		int64_t stride;
		torch::nn::Sequential downsample;

		torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr };
		torch::nn::BatchNorm2d bn1{ nullptr }, bn2{ nullptr }, bn3{ nullptr };

		static int expansion;

		Bottleneck(
			int64_t inplanes,
			int64_t planes,
			int64_t stride = 1,
			torch::nn::Sequential downsample = nullptr,
			int64_t groups = 1,
			int64_t base_width = 64,
			int64_t dilation = 1);

		torch::Tensor forward(torch::Tensor X);
	};
} // namespace _resnetimpl

template <typename Block>
struct ResNetImpl : torch::nn::Module {
	int64_t groups, base_width, inplanes;
	int64_t _dilation;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d bn1;
	torch::nn::Functional max_pool1;
	torch::nn::Linear fc;
	torch::nn::Sequential layer1, layer2, layer3, layer4;
	bool hookfeature;


	torch::nn::Sequential _make_layer(
		int64_t planes,
		int64_t blocks,
		int64_t stride = 1,
		bool dilate = false);

	ResNetImpl(
		const std::vector<int>& layers,
		int64_t num_classes = 1000,
		bool zero_init_residual = false,
		int64_t groups = 1,
		int64_t width_per_group = 64);
	torch::Tensor forward(torch::Tensor X);
};

template <typename Block>
torch::nn::Sequential ResNetImpl<Block>::_make_layer(
	int64_t planes,
	int64_t blocks,
	int64_t stride,
	bool dilate) 
{
	int64_t previous_dilation = _dilation;


	if(dilate)
	{
		_dilation *= stride;
		stride = 1;
	}


	torch::nn::Sequential downsample = nullptr;
	if (stride != 1 || inplanes != planes * Block::expansion) {
		downsample = torch::nn::Sequential(
			_resnetimpl::conv1x1(inplanes, planes * Block::expansion, stride),
			torch::nn::BatchNorm2d(planes * Block::expansion));
	}

	torch::nn::Sequential layers;
	layers->push_back(
		Block(inplanes, planes, stride, downsample, groups, base_width, previous_dilation));

	inplanes = planes * Block::expansion;

	for (int i = 1; i < blocks; ++i)
		layers->push_back(Block(inplanes, planes, 1, nullptr, groups, base_width, _dilation));

	return layers;
}

template <typename Block>
ResNetImpl<Block>::ResNetImpl(
	const std::vector<int>& layers,
	int64_t num_classes,
	bool zero_init_residual,
	int64_t groups,
	int64_t width_per_group)
	: groups(groups),
	base_width(width_per_group),
	inplanes(64),
	conv1(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(
		false).dilation(1)),
	bn1(64),
	fc(512 * Block::expansion, num_classes),
	max_pool1(torch::max_pool2d, 3, 2, 1, 1, false) {

	_dilation = 1;

	layer1 = _make_layer(64, layers[0]);
	layer2 = _make_layer(128, layers[1], 2);
	layer3 = _make_layer(256, layers[2], 2, true);
	layer4 = _make_layer(512, layers[3], 2, true);

	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("max_pool1", max_pool1);
	register_module("fc", fc);

	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);

	for (auto& module : modules(/*include_self=*/false)) {
		if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
			torch::nn::init::kaiming_normal_(
				M->weight,
				/*a=*/0,
				torch::kFanOut,
				torch::kReLU);
		else if (auto M = dynamic_cast<torch::nn::BatchNormImpl*>(module.get())) {
			torch::nn::init::constant_(M->weight, 1);
			torch::nn::init::constant_(M->bias, 0);
		}
	}

	// Zero-initialize the last BN in each residual branch, so that the residual
	// branch starts with zeros, and each residual block behaves like an
	// identity. This improves the model by 0.2~0.3% according to
	// https://arxiv.org/abs/1706.02677
	if (zero_init_residual)
		for (auto& module : modules(/*include_self=*/false)) {
			if (auto * M = dynamic_cast<_resnetimpl::Bottleneck*>(module.get()))
				torch::nn::init::constant_(M->bn3->weight, 0);
			else if (auto * M = dynamic_cast<_resnetimpl::BasicBlock*>(module.get()))
				torch::nn::init::constant_(M->bn2->weight, 0);
		}
}

template <typename Block>
torch::Tensor ResNetImpl<Block>::forward(torch::Tensor x) {
	x = conv1->forward(x);
	x = bn1->forward(x).relu_();
	x = max_pool1->forward(x);

	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);

	x = torch::adaptive_avg_pool2d(x, { 1, 1 });
	x = x.view({ x.sizes()[0], -1 });
	x = fc->forward(x);

	return x;
}

struct ResNet18Impl : ResNetImpl<_resnetimpl::BasicBlock> {
	ResNet18Impl(int64_t num_classes = 1000, bool zero_init_residual = false);
};

struct ResNet34Impl : ResNetImpl<_resnetimpl::BasicBlock> {
	ResNet34Impl(int64_t num_classes = 1000, bool zero_init_residual = false);
};

struct ResNet50Impl : ResNetImpl<_resnetimpl::Bottleneck> {
	ResNet50Impl(int64_t num_classes = 1000, bool zero_init_residual = false);
};

struct ResNet101Impl : ResNetImpl<_resnetimpl::Bottleneck> {
	ResNet101Impl(int64_t num_classes = 1000, bool zero_init_residual = false);
};

struct ResNet152Impl : ResNetImpl<_resnetimpl::Bottleneck> {
	ResNet152Impl(int64_t num_classes = 1000, bool zero_init_residual = false);
};

struct ResNext50_32x4dImpl : ResNetImpl<_resnetimpl::Bottleneck> {
	ResNext50_32x4dImpl(
		int64_t num_classes = 1000,
		bool zero_init_residual = false);
};

struct ResNext101_32x8dImpl : ResNetImpl<_resnetimpl::Bottleneck> {
	ResNext101_32x8dImpl(
		int64_t num_classes = 1000,
		bool zero_init_residual = false);
};

template <typename Block>
struct ResNet : torch::nn::ModuleHolder<ResNetImpl<Block>> {
	using torch::nn::ModuleHolder<ResNetImpl<Block>>::ModuleHolder;
};

TORCH_MODULE(ResNet18);
TORCH_MODULE(ResNet34);
TORCH_MODULE(ResNet50);
TORCH_MODULE(ResNet101);
TORCH_MODULE(ResNet152);
TORCH_MODULE(ResNext50_32x4d);
TORCH_MODULE(ResNext101_32x8d);

#endif // RESNET_H
