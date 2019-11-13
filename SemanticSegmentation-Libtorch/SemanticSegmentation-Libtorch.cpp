#include <iostream>
#include "cocoapi/cocoNote.h"

int main()
{
	COCONote test;
	test.Parse("annotations/instances_val2017.json");
}
