#include "phase_graycoding.hpp"
#include "fringe_analysis.hpp"
#include "graycoding.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cmath>

using namespace std;
using namespace cv;

void phaseGraycodingUnwrap(const vector<string>& imlist_ps, const vector<string>& imlist_gc,
	OutputArray _Phi, int p, int N)
{
	// Estimate wrapped phase map
	Mat1d phi;
	NStepPhaseShifting(imlist_ps, phi, N);

	// Estimate code words
	Mat code_word;
	codeword(imlist_gc, code_word);

	// Estimate fringe order with codeword
	Mat1d k = grayToDec(code_word);

	// Shift and rewrap wrapped phase
	double shift = -CV_PI + CV_PI / p;
	double* phid = (double*)phi.data;
	for (int i = 0; i < phi.total(); i++)
		phid[i] = atan2(sin(phid[i] + shift), cos(phid[i] + shift));

	// Estimate absolute phase map
	_Phi.create(phi.size(), CV_64F);
	Mat1d Phi = _Phi.getMat();
	Phi = phi + 2 * CV_PI * k;

	// Shift phase back to the original values
	Phi -= shift;

	// Filter spiky noise
	Mat1f Phim;
	medianBlur((Mat_<float>)Phi, Phim, 5);

	double* pPhi = (double*)Phi.data;
	float* pPhim = (float*)Phim.data;
	for (int i = 0; i < Phi.total(); i++)
		pPhi[i] -= 2 * CV_PI * cvRound((pPhi[i] - pPhim[i]) / 2 / CV_PI);
}