/*
 * MultiperceptronNeuralNetwork.h
 *
 *  Created on: Mar 3, 2019
 *      Author: mchrusci
 */

#include <vector>
#include <Eigen/Dense>

#ifndef MULTIPERCEPTRONNEURALNETWORK_H_
#define MULTIPERCEPTRONNEURALNETWORK_H_

class MultiperceptronNeuralNetwork {
public:
	MultiperceptronNeuralNetwork(Eigen::MatrixXf inputMatrix,
								 std::vector<short> hiddenLayersNeuronsNum,
								 Eigen::MatrixXf outputMatrix);
	virtual ~MultiperceptronNeuralNetwork();

	// Getter
	std::vector<Eigen::MatrixXf> getWeights() const;
	std::vector<Eigen::VectorXf> getBiases() const;

private:
	std::vector<Eigen::MatrixXf> weightsMatrices;
	std::vector<Eigen::VectorXf> biasesVectors;
};

#endif /* MULTIPERCEPTRONNEURALNETWORK_H_ */
