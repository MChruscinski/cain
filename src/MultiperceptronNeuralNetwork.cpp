/*
 * MultiperceptronNeuralNetwork.cpp
 *
 *  Created on: Mar 3, 2019
 *      Author: mchrusci
 */

#include "MultiperceptronNeuralNetwork.h"

MultiperceptronNeuralNetwork::MultiperceptronNeuralNetwork(
									Eigen::MatrixXf inputMatrix,
									std::vector<short> hiddenLayersNeuronsNum,
									Eigen::MatrixXf outputMatrix)
{
	const short numberOfLayers = hiddenLayersNeuronsNum.size() + 2;

	std::vector<short> layersNeuronsNum;
	layersNeuronsNum.reserve(numberOfLayers);
	layersNeuronsNum.push_back(inputMatrix.rows());
	layersNeuronsNum.insert(layersNeuronsNum.end(), hiddenLayersNeuronsNum.begin(),
			hiddenLayersNeuronsNum.end());
	layersNeuronsNum.push_back(outputMatrix.rows());

	std::vector<Eigen::MatrixXf> weightsMatrices(numberOfLayers-1);
	std::vector<Eigen::VectorXf> biasesVectors(numberOfLayers-1);

	for(auto layersNeuronsNum_Iter = layersNeuronsNum.begin() + 1;
			layersNeuronsNum_Iter != layersNeuronsNum.end();
			++layersNeuronsNum_Iter)
	{
		weightsMatrices.push_back(Eigen::MatrixXf::Random(*layersNeuronsNum_Iter,
													*(layersNeuronsNum_Iter-1)));
		biasesVectors.push_back(Eigen::VectorXf::Random(*layersNeuronsNum_Iter));
	}

	this->weightsMatrices = weightsMatrices;
	this->biasesVectors = biasesVectors;
}

MultiperceptronNeuralNetwork::~MultiperceptronNeuralNetwork() {
	// TODO Auto-generated destructor stub
}

// Getters
std::vector<Eigen::MatrixXf> MultiperceptronNeuralNetwork::getWeights() const
{
	return this->weightsMatrices;
}

std::vector<Eigen::VectorXf> MultiperceptronNeuralNetwork::getBiases() const
{
	return this->biasesVectors;
}
