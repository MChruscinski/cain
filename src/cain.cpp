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
	std::vector<float> input{1,2,3,4};
	std::vector<short> hiddenLayersNeuronsNum{2,3};
	const short outputLength = 1;

	// **************************************************************************

	auto* model = new MultiperceptronNeuralNetwork(input,
											hiddenLayersNeuronsNum,
											outputLength);


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
