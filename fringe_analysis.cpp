#include "fringe_analysis.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cmath>

using namespace std;
using namespace cv;

void NStepPhaseShifting(const vector<string>& imgs, OutputArray _phase, int N)
{
	double delta = 2 * CV_PI / N;
	Mat1d I = imread(imgs[0], 0);
	Mat1d sumIsin = I * sin(delta);
	Mat1d sumIcos = I * cos(delta);

	for (int i = 1; i < imgs.size(); i++)
	{
		Mat1d I = imread(imgs[i], 0);
		double delta = 2 * CV_PI * (i + 1) / N;
		sumIsin += I * sin(delta);
		sumIcos += I * cos(delta);
	}

	_phase.create(sumIsin.size(), sumIsin.type());
	Mat1d phase = _phase.getMat();

	double* pphase = (double*)phase.data;
	double* psumIsin = (double*)sumIsin.data;
	double* psumIcos = (double*)sumIcos.data;
	for (int i = 0; i < sumIsin.total(); i++)
		pphase[i] = -atan2(psumIsin[i], psumIcos[i]);
}

void ThreeStepPhaseShifting(const vector<string>& imgs, OutputArray _phase)
{
	vector<Mat1d> I(3);
	for (int i = 0; i < 3; i++)
		I[i] = imread(imgs[i], 0);

	_phase.create(I[0].size(), I[0].type());
	Mat1d phase = _phase.getMat();
	for (int i = 0; i < phase.rows; i++)
		for (int j = 0; j < phase.cols; j++)
			phase(i, j) = atan2(sqrt(3) * (I[0](i, j) - I[2](i, j)), 2 * I[1](i, j) - I[0](i, j) - I[2](i, j));
}

//void modulation(cv::InputArray, cv::OutputArray, int);