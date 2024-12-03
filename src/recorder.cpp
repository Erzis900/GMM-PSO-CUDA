#include "recorder.h"
#include <stdio.h>

Recorder::Recorder(void)
{
	delay = 1;
	delay_iterator = delay;
}

Recorder::~Recorder(void)
{
	if (!container.empty())
		container.clear();
}

/// set delay
void Recorder::setDelay(int value)
{
	delay = value;
	delay_iterator = delay;
}

/// save point
void Recorder::save(float point)
{
	if (delay_iterator == delay)
	{
		container.push_back(point);
		delay_iterator = 1;
	}
	else
		delay_iterator++;
}

/// save to file
void Recorder::save2file(const char *filename, const char *var_name)
{
	FILE *plik;
	std::list<float>::iterator it;

	plik = fopen(filename, "w+t");
	fprintf(plik, "%s=[", var_name);
	for (it = container.begin(); it != container.end(); it++)
	{
		fprintf(plik, "%.4f,", (*it));
	}
	fprintf(plik, "];\n");
	fprintf(plik, "plot(%s);\n", var_name);

	fclose(plik);
}

/// clear container
void Recorder::clearContainer(void)
{
	container.clear();
	delay_iterator = delay;
}
