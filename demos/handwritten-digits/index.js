const
  mnist = require( 'mnist' ),
  commandLineArgs = require( 'command-line-args' ),
  options = commandLineArgs([
    { name: 'train', alias: 't', type: Number, multiple: true },
    { name: 'proof', alias: 'p', type: String, multiple: true },
    { name: 'src', type: String, multiple: true, defaultOption: true }
  ]),
  {
    makeLayerWithRandomParameters,
    trainNeuralNetworkEpochs,
    runNeuralNetwork
  } = require( '../../src/neural-network' ),
  {
    logEpoch,
    logFinalResult,
    getIndexOfBiggestNumberInArray,
    showImage,
    loadParameters,
    saveParameters
  } = require( './helpers' );

function checkNeuralNetworkAccuracy( parameters, trainingData ) {
  let
    success = 0,
    failure = 0,
    total = 0;

  trainingData.forEach( trainingDataPiece => {
    const
      neuralOutput = getIndexOfBiggestNumberInArray( runNeuralNetwork( trainingDataPiece.input, parameters )),
      expectedOutput = getIndexOfBiggestNumberInArray( trainingDataPiece.output );

    if ( neuralOutput === expectedOutput ) {
      success++;
    } else {
      failure++;
    }
    total++;
  });

  return { success, failure, total };
}

function trainNeuralNetworkOverEpochs( trainingSet, testingSet, epochs, learningRate ) {
  const
    parameters = [
      makeLayerWithRandomParameters( 16, 784 ),
      makeLayerWithRandomParameters( 16, 16 ),
      makeLayerWithRandomParameters( 10, 16 )
    ],
    startTime = Date.now(),
    logProgress = ( adjustedParams, epochIndex ) => {
      const accuracy = checkNeuralNetworkAccuracy( adjustedParams, testingSet.slice( 0, 100 ));

      logEpoch( epochIndex, Date.now() - startTime, accuracy );
    },
    trainedParameters = trainNeuralNetworkEpochs( trainingSet, parameters, learningRate, epochs, logProgress ),
    accuracyResult = checkNeuralNetworkAccuracy( trainedParameters, testingSet ),
    savedPath = saveParameters(
      parameters,
      trainingSet.length,
      testingSet.length,
      accuracyResult.success / accuracyResult.total,
      epochs
    );

  logFinalResult( accuracyResult, savedPath );
}

function proofNeuralNetwork( parametersPath ) {
  const
    randomDigit = mnist.get( 1 ),
    input = randomDigit[ 0 ].input,
    parameters = loadParameters( parametersPath ),
    startTime = Date.now(),
    result = getIndexOfBiggestNumberInArray( runNeuralNetwork( input, parameters )),
    timeElapsed = Date.now() - startTime;

  showImage( input, result, timeElapsed );
}

function runTraining( trainingSetSize, testingSetSize, epochs ) {
  const { training: trainingSet, test: testingSet } = mnist.set( trainingSetSize, testingSetSize );

  trainNeuralNetworkOverEpochs( trainingSet, testingSet, epochs, 1 );
}

if ( options.train != null ) {
  const [ trainingSetSize = 1000, testingSetSize = 100, epochs = 30 ] = options.train;

  runTraining( trainingSetSize, testingSetSize, epochs );
} else if ( options.proof !== null ) {
  const [ parametersPath, timeout = 3000 ] = options.proof;

  setInterval(() => proofNeuralNetwork( parametersPath ), timeout );
}
