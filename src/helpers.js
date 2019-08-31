function sumArraysItems( arrayA, arrayB ) {
  return arrayA.map(( item, index ) => item + arrayB[ index ]);
}

function averageManyArraysItems( arrays ) {
  const
    arraysCount = arrays.length,
    arraysSize = arrays[ 0 ].length,
    zeroedArray = ( new Array( arraysSize )).fill( 0, 0, arraysSize );

  return arrays
    .reduce(( sumArray, array ) => sumArraysItems( sumArray, array ), zeroedArray )
    .map( item => item / arraysCount );
}

function subtractArrayItems( arrayA, arrayB ) {
  return arrayA.map(( item, index ) => item - arrayB[ index ]);
}

function multiplyArrayItemsByScalar( scalarNumber, array ) {
  return array.map( item => scalarNumber * item );
}

function averageOfArray( array ) {
  return array.reduce(( sum, item ) => ( sum + item ), 0 ) / array.length;
}

function lastArrayItem( array ) {
  return array[ array.length - 1 ];
}

function randomNumber( lower, higher ) {
  return Math.random() * ( higher - lower ) + lower;
}

module.exports = {
  averageManyArraysItems,
  subtractArrayItems,
  multiplyArrayItemsByScalar,
  lastArrayItem,
  randomNumber
};
