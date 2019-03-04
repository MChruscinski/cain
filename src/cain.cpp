//============================================================================
// Name        : cain.cpp
// Author      : Michal Chruscinski
// Version     :
// Copyright   : GNU GENERAL PUBLIC LICENSE
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include "MultiperceptronNeuralNetwork.h"

int main() {

	std::cout << "CAIN!" << std::endl; // prints CAIN!

	// **************************************************************************
	// VARIABLES THAT WILL EVENTUALLY BECOME USER DEFINED
	// AND AN INTERFACE I GUESS OR WHATEVER
	//const short inputLength = 4; // will eventually be determined automagically but whatever
														       // ^ its toally a word frak you
	const short datasetSize = 100;
	auto inputMatrix = Eigen::MatrixXf::Random(4, datasetSize);
	std::vector<short> hiddenLayersNeuronsNum{2,3};
	auto outputMatrix = Eigen::MatrixXf::Random(2, datasetSize);

	// **************************************************************************

	auto* model = new MultiperceptronNeuralNetwork(inputMatrix,
											hiddenLayersNeuronsNum,
											outputMatrix);


	for(auto weightsMatrix : model->getWeights())
	{
		std::cout << weightsMatrix << std::endl << std::endl;
	}
	for (auto biasVector : model->getBiases())
	{
		std::cout << biasVector << std::endl << std::endl;
	}

	return 0;
}
