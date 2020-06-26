#include "graycoding.hpp"
#include "utils.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cmath>

using namespace std;
using namespace cv;

Mat1i grayToDec(InputArray _code_word)
{
	Mat code_word = _code_word.getMat();
	Mat1i dec(code_word.size[0], code_word.size[1], 0);

	int h = code_word.size[0], w = code_word.size[1], n = code_word.size[2];

	uchar* pcode_word = (uchar*)code_word.data;
	int* pdec = (int*)dec.data;
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
			int dec_ = 0;
			uchar tmp = pcode_word[i*w + j];
			if (tmp)
				dec_ += 1 << (n - 1);

			for (int k = 1; k < n; k++)
			{
				tmp ^= pcode_word[i*w + j + k*w*h];
				if (tmp)
					dec_ += 1 << (n - k - 1);
			}

			pdec[i*w + j] = dec_;
		}

	return dec;
}

void codeword(const vector<string>& imlist, OutputArray _code_word)
{
	// Gruping images as pairs
	vector<vector<string>> l = chunks(imlist, 2);

	// Read first image for output array size
	Size sz = imread(l[0][0], 0).size();

	// Setting output 3D array
	int w = sz.width, h = sz.height;
	int dims[] = { h, w, static_cast<int>(l.size()) };
	_code_word.create(3, dims, CV_8U);
	Mat code_word = _code_word.getMat();

	uchar* pcode_word = (uchar*)code_word.data;
	for (int k = 0; k < l.size(); k++)
	{
		Mat1b im1 = imread(l[k][0], 0);
		Mat1b im2 = imread(l[k][1], 0);
		Mat1b bin = (im1 > im2) / 255;
		uchar* pbin = (uchar*)bin.data;
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
				pcode_word[i*w + j + k*w*h] = pbin[i*w + j];
	}
}

void decode(InputArray _code_word, vector<float>& coor, InputArray _mask)
{
	Mat1i dec = grayToDec(_code_word);
	Mat1b mask = _mask.getMat();
	coor.reserve(countNonZero(mask));

	uchar* pmask = (uchar*)mask.data;
	int* pdec = (int*)dec.data;
	for (int i = 0; i < mask.total(); i++)
		if (pmask[i])
			coor.push_back(static_cast<float>(pdec[i]));
}
