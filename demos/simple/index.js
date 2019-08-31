const {
    makeLayerWithRandomParameters,
    trainNeuralNetworkEpochs,
    runNeuralNetwork
  } = require( '../../src/neural-network' );

// creating the problem
function isFirstBiggerThanSecondSquared( numberA, numberB ) {
  return numberA > Math.pow( numberB, 2 ) ? 1 : 0;
}

function makeRandomNumbersPairs( amountToCreate ) {
  const numbers = [];

  for ( let i = 0; i < amountToCreate; i++ ) { // for since recursion hits maximum call stack size with 10k
    numbers.push([ Math.random(), Math.random() ]);
  }

  return numbers;
}

function makeTrainingSet( amount ) {
  return makeRandomNumbersPairs( amount )
    .map( numberPair => ({
      input: numberPair,
      output: [ isFirstBiggerThanSecondSquared( numberPair[ 0 ], numberPair[ 1 ]) ]
    }));
}

const
  trainingSet = makeTrainingSet( 10000 ),
  testingSet = makeTrainingSet( 1000 );

// solving the problem
const
  parameters = [ makeLayerWithRandomParameters( 1, 2 ) ], // shapes the neural network
  trainedParameters = trainNeuralNetworkEpochs( trainingSet, parameters, 1, 10 ), // trains the neural network
  successes = testingSet.reduce(( successSum, { input, output }) => {
    const neuralNetWorkOutput = runNeuralNetwork( input, trainedParameters ) > 0.5 ? 1 : 0; // uses the neural network

    if ( neuralNetWorkOutput === output[ 0 ]) {
      return successSum + 1;
    }

    return successSum;
  }, 0 );

console.log( `Successes: ${ successes }, Total tests: ${ testingSet.length }` );
