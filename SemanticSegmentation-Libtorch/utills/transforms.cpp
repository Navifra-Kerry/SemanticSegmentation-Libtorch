#include "transforms.h"
namespace transforms {
namespace polygon {
void Resize(double scaleX, double scaleY, std::vector<std::vector<double>>& polygon)
{
	int j = 0;
	for (int idx = 0; idx < polygon.size(); idx++)
	{
		double x = 0, y = 0;
		double* xy = polygon[idx].data();

		for (j = 0; j < polygon[idx].size() / 2; j++) x += (xy[j * 2 + 0]);
		for (j = 0; j < polygon[idx].size() / 2; j++) y += (xy[j * 2 + 1]);

		for (j = 0; j < polygon[idx].size() / 2; j++) xy[j * 2 + 0] = (xy[j * 2 + 0] * scaleX);
		for (j = 0; j < polygon[idx].size() / 2; j++) xy[j * 2 + 1] = (xy[j * 2 + 1] * scaleY);
	}
}
}//transforms
}//polygon