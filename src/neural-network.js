const
  {
    derivativeOfCostWithRespectToWeight,
    derivativeOfCostWithRespectToBias,
    derivativeOfCostWithRespectToInput,
    sigmoid: activationFunction
  } = require( './math' ),
  {
    averageManyArraysItems,
    subtractArrayItems,
    multiplyArrayItemsByScalar,
    lastArrayItem,
    randomNumber
  } = require( './helpers' );

function aggregateValues( inputValues, weights, bias ) {
  return weights.reduce(( sum, weight, index ) => sum + weight * inputValues[ index ], 0 ) + bias;
}

function computeNeuron( inputValues, parameters ) {
  const
    { weights, bias } = parameters,
    aggregateValue = aggregateValues( inputValues, weights, bias ),
    activationValue = activationFunction( aggregateValue );

  return { aggregateValue, activationValue };
}

function computeNeuralNetwork( networkActivations, parameters, layerIndex = 0, networkAggregates = []) {
  const
    inputValues = lastArrayItem( networkActivations ),
    layerParameters = parameters[ layerIndex ],
    layerNeuronsOutput = layerParameters.map( neuronParameters => computeNeuron( inputValues, neuronParameters )),
    layerAggregates = layerNeuronsOutput.map(({ aggregateValue }) => aggregateValue ), // pluck from every output
    layerActivations = layerNeuronsOutput.map(({ activationValue }) => activationValue ),
    newNetworkActivations = networkActivations.concat([ layerActivations ]),
    newNetworkAggregates = networkAggregates.concat([ layerAggregates ]),
    nextLayerIndex = layerIndex + 1;

  if ( nextLayerIndex === parameters.length ) { // next layer won't exist
    return { networkActivations: newNetworkActivations, networkAggregates: newNetworkAggregates };
  }

  return computeNeuralNetwork( newNetworkActivations, parameters, nextLayerIndex, newNetworkAggregates );
}

function computeNeuronGradients( layerInputs, neuronParameters, aggregateValue, activationValue, expectedValue ) {
  const
    inputsGradient = neuronParameters.weights.map( weight =>
      derivativeOfCostWithRespectToInput( weight, aggregateValue, activationValue, expectedValue )
    ),
    weightsGradient = neuronParameters.weights.map(( weight, index ) =>
      derivativeOfCostWithRespectToWeight( layerInputs[ index ], aggregateValue, activationValue, expectedValue )
    ),
    biasGradient = derivativeOfCostWithRespectToBias( aggregateValue, activationValue, expectedValue );

  return { inputsGradient, weightsGradient, biasGradient };
}

function computeLayerGradients(
  layerInputs,
  layerParameters,
  layerAggregations,
  layerActivations,
  layerExpectedActivations
) {
  return layerParameters.map(( neuronParameters, neuronIndex ) => {
    const
      expectedValue = layerExpectedActivations[ neuronIndex ],
      aggregateValue = layerAggregations[ neuronIndex ],
      activationValue = layerActivations[ neuronIndex ];

    return computeNeuronGradients( layerInputs, neuronParameters, aggregateValue, activationValue, expectedValue );
  });
}

function adjustLayerParams( layerParameters, layerGradients, learningRate ) {
  return layerParameters.map(( neuronParameters, neuronIndex ) => {
    const neuronGradients = layerGradients[ neuronIndex ];

    return {
      weights: computeAdjustedNeuronWeights( neuronParameters.weights, neuronGradients.weightsGradient, learningRate ),
      bias: computeAdjustedBias( neuronParameters.bias, neuronGradients.biasGradient, learningRate )
    };
  });
}

function adjustPreviousLayerActivations( previousLayerActivations, layerGradients, learningRate ) {
  const
    inputsGradients = layerGradients.map(({ inputsGradient }) => inputsGradient ),
    inputsMeanGradient = averageManyArraysItems( inputsGradients ); // maybe here is just sum

  return computeAdjustedPreviousActivations( previousLayerActivations, inputsMeanGradient, learningRate );
}

function adjustParameters(
  parameters,
  learningRate,
  networkAggregates,
  networkActivations,
  layerExpectedActivations,
  layerIndex,
  adjustedParameters = []
) {
  const
    layerParameters = parameters[ layerIndex ],
    layerInputs = networkActivations[ layerIndex ],
    layerActivations = networkActivations[ layerIndex + 1 ],
    layerAggregations = networkAggregates[ layerIndex ],
    layerGradients = computeLayerGradients(
      layerInputs,
      layerParameters,
      layerAggregations,
      layerActivations,
      layerExpectedActivations
    ),
    adjustedLayerParameters = adjustLayerParams( layerParameters, layerGradients, learningRate ),
    newAdjustedParameters = adjustedParameters.concat([ adjustedLayerParameters ]),
    previousLayerAdjustedActivations = adjustPreviousLayerActivations( layerInputs, layerGradients, learningRate ),
    previousLayerIndex = layerIndex - 1;

  if ( previousLayerIndex === -1 ) {
    return newAdjustedParameters.reverse();
  }

  return adjustParameters(
    parameters,
    learningRate,
    networkAggregates,
    networkActivations,
    previousLayerAdjustedActivations,
    previousLayerIndex,
    newAdjustedParameters
  );
}

function computeAdjustedNeuronWeights( neuronWeights, neuronWeightGradient, learningRate ) {
  return subtractArrayItems(
    neuronWeights,
    multiplyArrayItemsByScalar( learningRate, neuronWeightGradient )
  );
}

function computeAdjustedPreviousActivations( previousLayerActivations, inputsMeanGradient, learningRate ) {
  return subtractArrayItems(
    previousLayerActivations,
    multiplyArrayItemsByScalar( learningRate, inputsMeanGradient )
  );
}

function computeAdjustedBias( neuronBias, neuronBiasGradient, learningRate ) {
  return neuronBias - learningRate * neuronBiasGradient;
}

/**
 * Run the neural network calculations and returns the activations of the last layer.
 * @param {Array} input
 * @param {Array} parameters
 * @return {Array} neural network output
 */
function runNeuralNetwork( input, parameters ) {
  return lastArrayItem( computeNeuralNetwork([ input ], parameters ).networkActivations );
}

function generateRandomWeights( amount, weights = []) {
  if ( amount === 0 ) {
    return weights;
  }

  const weight = randomNumber( -1, 1 );

  return generateRandomWeights( amount - 1, weights.concat( weight ));
}

function generateRandomNeuronsParameters( amountOfNeurons, amountOfWeights, neuronsParameters = []) {
  if ( amountOfNeurons === 0 ) {
    return neuronsParameters;
  }

  const neuronParameters = {
    bias: randomNumber( -1, 1 ),
    weights: generateRandomWeights( amountOfWeights )
  };

  return generateRandomNeuronsParameters(
    amountOfNeurons - 1,
    amountOfWeights,
    neuronsParameters.concat( neuronParameters )
  );
}

/**
 * Creates a layer of neuron parameters, which is basically an array of { weights: [], bias }.
 * @param amountOfNeurons
 * @param amountOfInputs
 * @return {Array}
 */
function makeLayerWithRandomParameters( amountOfNeurons, amountOfInputs ) {
  return generateRandomNeuronsParameters( amountOfNeurons, amountOfInputs );
}

/**
 * Trains the neural network for a single example.
 * @param {Array} inputValues
 * @param {Array} parameters
 * @param {Array} expectedValue
 * @param {Number} learningRate
 * @return {Array} adjusted parameters
 */
function trainNeuralNetwork( inputValues, parameters, expectedValue, learningRate = 1 ) {
  const { networkActivations, networkAggregates } = computeNeuralNetwork([ inputValues ], parameters );

  return adjustParameters(
    parameters,
    learningRate,
    networkAggregates,
    networkActivations,
    expectedValue,
    parameters.length - 1
  );
}

/**
 * Trains the neural network over the whole training set.
 * @param {Array} trainData
 * @param {Array} parameters
 * @param {number} learningRate
 * @return {Array} adjusted parameters
 */
function trainNeuralNetworkEpoch( trainData, parameters, learningRate ) {
  return trainData.reduce(( adjustedParams, { input, output }) =>
    trainNeuralNetwork( input, adjustedParams, output, learningRate ),
    parameters
  );
}

/**
 * Adjust the network params for the whole training set multiple times.
 * @param {Array} trainData
 * @param {Array} parameters
 * @param {number} learningRate
 * @param {number} epochsYetToRun
 * @param {function} onEpochFinish
 * @return {Array} adjusted parameters
 */
function trainNeuralNetworkEpochs(
  trainData,
  parameters,
  learningRate,
  epochsYetToRun,
  onEpochFinish
) {
  if ( epochsYetToRun === 0 ) {
    return parameters;
  }

  const newParameters = trainNeuralNetworkEpoch( trainData, parameters, learningRate );

  if ( typeof onEpochFinish === 'function' ) {
    onEpochFinish( newParameters, epochsYetToRun );
  }

  return trainNeuralNetworkEpochs( trainData, newParameters, learningRate, epochsYetToRun - 1, onEpochFinish );
}

module.exports = {
  makeLayerWithRandomParameters,
  trainNeuralNetworkEpochs,
  trainNeuralNetworkEpoch,
  trainNeuralNetwork,
  runNeuralNetwork
};
