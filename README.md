# Boto Neural Network

This is a JavaScript side project created in a week on spare time with the intent to learn 
how neural networks works on the inside. The kind of neural network implemented here 
is the online gradient descent using sigmoid as activation function. It 
was created trying to follow the functional programming principles, but 
also without getting too fancy or using external libraries to handle it. 

Boto is also a kind of dolphin that [ helps fisherman to catch fish ]( https://www.livescience.com/20027-dolphins-work-fishermen.html ).
Dolphins are well known for their intelligence as well as neural networks are. 
Helping humans on their daily tasks instead of addicting them on social networks of all sorts 
and displaying personalized and intrusive ads is something machine learning needs to learn with botos.
With machine learning reshaping how we live and work the question inevitable to ask is:
are we coding shark machines or boto machines?

## Recognizing handwritten digits

![boto-proof](https://user-images.githubusercontent.com/6374422/64067375-ecb59b80-cbfd-11e9-899f-92ffc8467d67.gif)

This seems to be the most classical initial project on neural networks and it was quite fun to do and see it working.
The trained parameters on `demos/handwritten-digits/parameters` provides a 97.6% accuracy, 
to get there it ran for 3 hours with 8000 training set. A next step will be to train it with 60k images but will 
probably do it on the cloud.

### See it alive 

Clone the project and:
```
npm run proof
npm run proof { path to parameters file } { time between proofs in ms }
```

### Train it 
```
npm run train
npm run train { training set size } { testing set size } { amount of epochs }
```

### Extend it

Checkout the `demos/handwritten-digis` folder, there is the source files for this 
handwritten recognition implementation. 


Important to mention that this is a demo of the project, not the project itself. The neural network is what is
inside the `src` folder and the main file is `src/neural-network.js`.

## Use it on your projects

You can install it using npm:
```
npm install boto-neural-network
```

A simple demonstration:

```js
const {
    makeLayerWithRandomParameters,
    trainNeuralNetworkEpochs,
    runNeuralNetwork
  } = require( 'boto-neural-network' );

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

    if ( neuralNetWorkOutput === output[ 0 ] ) {
      return successSum + 1;
    }

    return successSum;
  }, 0 );

console.log( `Successes: ${ successes }, Total tests: ${ testingSet.length }` );

```
You can find this working code on `demos/simple/index.js` and can run it like this: `npm run simple`.

This sample code runs in less than a second and gives a success ratio greater than between 96% and 98%.
You can play with this example trying to add more layers and/or to make the problem more complex or simpler.



## API

This package provides basically 3 simple functions:

### makeLayerWithRandomParameters()
It creates a layer of neuron parameters, which is basically an array of `{ weights: [], bias }`. 
Important to note that a combination of the layers returned by this function shapes the neural network.

#### Arguments

`amountOfNeurons`: The amount of neurons a layer will contain.

`amountOfInputs`: The amount of inputs each neuron will take, on the first layer it is the amount of 
input you have, on the following it should be the amount of neurons from the previous layer.

#### Return

An array of `{ weights: [], bias }`

### trainNeuralNetworkEpochs()
This function basically takes an set of inputs, expected outputs and parameters and returns a new set of
parameters, as all train functions on this project, the difference is that it does it for the whole training set
multiple times. This is the most straight forward function to train a network but you can get fancier,
like adjusting the learning rate on every epoch, by using the other train functions.

One important thing to note here, since the project tried to follow functional programming principles 
if you set 10k epochs it will crash given the maximum call stack size, if you need that many epochs 
for whatever reason consider using `trainNeuralNetworkEpoch` inside a normal loop. 

#### Arguments

`trainData`: The training set, an array of objects like: `{ input: [], output: [] }`. Note that the input and 
output needs to arrays.

`parameters`: The parameters to start with, it is an array of layers ( an array of 
`makeLayerWithRandomParameters` outputs ). This is what defines the shape of the neural network.

`learningRate`: This can make each train cycle to change the parameters more or less, there is no right value,
you need to play with it but starting with 1 is a good place.

`epochsYetToRun`: The amount of epochs to run, this will be used internally to limit the function recursion.

`onEpochFinish`: A function that will run after every epoch, the function takes `newParameters` and `epochIndex` 
as arguments so you can keep track of you network and also do sample test with the new parameters to see how it is
evolving.

#### Return

An new set of parameters, with error reduced.

### runNeuralNetwork()
This is the function that once you have trained parameters can be used to give the answers you need. Basically it
runs the neural network calculations over the layers and returns the activations of the last layer.

#### Arguments
`input`: One input, note that a single input is an Array with the input values.

`parameters`: The parameters to perform the calculations.

#### Return
The neural network output, which is an Array, even if the last layer contains only one neuron.

### trainNeuralNetworkEpoch()
Trains the neural network over the whole training set, but just once.

#### Arguments
`trainData`: The training set, as in `trainNeuralNetworkEpochs`.

`parameters`: The parameters to perform the calculations.

`learningRate`: The learning rate.

#### Return
An new set of parameters, with error reduced.

### trainNeuralNetwork()
Trains the neural network for a single example. 

#### Arguments
`inputValues`: A single example input, which is an Array with many input numbers.

`parameters`: The parameters to perform the calculations.

`expectedValue`: A single example out out, which is an Array with one or many numbers.

`learningRate`: The learning rate.

#### Return
An new set of parameters, with error reduced.

## Resources that contributed to this project
Workshop on 
[ BrazilJS ]( https://github.com/braziljs ) 
by 
[ MPJ ]( https://github.com/mpj )

Chapter 1 on 
[ Neural Networks and Deep Learning ]( http://neuralnetworksanddeeplearning.com/chap1.html ) 
by [ Michael Nielsen ]( https://github.com/mnielsen )

Youtube 
[ video series ]( https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2&t=0s )
by [ 3BLUE1BROWN ]( https://github.com/3b1b )


