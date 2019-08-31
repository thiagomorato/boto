function derivativeOfCostWithRespectToWeight( inputValue, aggregateValue, activationValue, expectedValue ) {
  return derivativeOfAggregationWithRespectToWeight( inputValue ) *
    derivativeOfActivationWithRespectToAggregation( aggregateValue ) *
    derivativeOfCostWithRespectToActivation( activationValue, expectedValue );
}

function derivativeOfCostWithRespectToBias( aggregateValue, activationValue, expectedValue ) {
  return derivativeOfAggregationWithRespectToBias() *
    derivativeOfActivationWithRespectToAggregation( aggregateValue ) *
    derivativeOfCostWithRespectToActivation( activationValue, expectedValue );
}

function derivativeOfCostWithRespectToInput( weight, aggregateValue, activationValue, expectedValue ) {
  return derivativeOfAggregationWithRespectToInput( weight ) *
    derivativeOfActivationWithRespectToAggregation( aggregateValue ) *
    derivativeOfCostWithRespectToActivation( activationValue, expectedValue );
}

function derivativeOfAggregationWithRespectToWeight( input ) {
  return input;
}

function derivativeOfActivationWithRespectToAggregation( aggregateValue ) {
  return sigmoidDerivative( aggregateValue );
}

function derivativeOfCostWithRespectToActivation( activationValue, expectedValue ) {
  return 2 * ( activationValue - expectedValue );
}

function derivativeOfAggregationWithRespectToBias() {
  return 1;
}

function derivativeOfAggregationWithRespectToInput( weight ) {
  return weight;
}

function sigmoid( number ) {
  return 1 / ( 1 + Math.exp( -number ));
}

function sigmoidDerivative( number ) {
  return sigmoid( number ) * ( 1 - sigmoid( number ));
}

module.exports = {
  derivativeOfCostWithRespectToWeight,
  derivativeOfCostWithRespectToBias,
  derivativeOfCostWithRespectToInput,
  sigmoid
};
