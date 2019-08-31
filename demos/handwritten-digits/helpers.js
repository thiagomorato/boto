const
  chalk = require( 'chalk' ),
  path = require( 'path' ),
  fs = require( 'fs' );

function loadParameters( parametersPath ) {
  const
    fallbackPath = path.join(
      __dirname,
      '/',
      'parameters',
      '/',
      'trained-parameters-train-set-8000-test-set-2000-accuracy-97.60.json',
    ),
    json = fs.readFileSync( parametersPath || fallbackPath );

  return JSON.parse( json );
}

function saveParameters( parameters, trainingSetSize, testingSetSize, accuracy, epochs ) {
  const
    timestamp = Date.now(),
    accuracyRatio = ( accuracy * 100 ).toFixed( 2 ),
    filePath = path.join(
      __dirname,
      '/parameters/',
      `${ timestamp }-training-${ trainingSetSize }` +
      `-test-${ testingSetSize }-${ epochs }-accuracy-${ accuracyRatio }.json`,
    );

  fs.writeFileSync(
    filePath,
    JSON.stringify( parameters )
  );

  return filePath;
}

function showImage( imageData, result, timeElapsed ) {
  console.log();
  for ( let top = 0; top < 28; top++ ) {
    let line = '';

    for ( let left = 0; left < 28; left++ ) {
      const value = Math.round( 255 * imageData[ top * 28 + left ]);

      line += chalk
        .bgRgb( value, value, value )
        .rgb( value, value, value )( String.fromCharCode( 9604 ) + String.fromCharCode( 9604 ));
    }

    if ( top === 2 && result != null ) {
      line += `   Neural network result: ${ chalk.green( result ) }`;
    }

    if ( top === 3 && timeElapsed != null ) {
      line += `   Processing time:       ${ timeElapsed } ms`;
    }

    console.log( line );
  }
  console.log();
}

function getIndexOfBiggestNumberInArray( expectedOutput ) {
  return expectedOutput.reduce(( biggestItem, number, index ) => {
    if ( number > biggestItem.number ) {
      return { number, index };
    }

    return biggestItem;
  }, { number: 0, index: 0 }).index;
}

function logEpoch( epochIndex, timeElapsed, accuracy ) {
  console.log(
    `Success rate: ${ chalk.green( `${ ( 100 * ( accuracy.success / accuracy.total )).toFixed( 2 ) }%` ) }, ` +
    `Remaining epochs: ${ chalk.yellow( epochIndex ) }, ` +
    `Time elapsed: ${ chalk.yellow( `${ timeElapsed }ms` ) }`
  );
}

function logFinalResult( accuracyResult, savedPath ) {
  console.log();
  console.log( chalk.red( '┌──────────────────┐' ));
  console.log( chalk.red( '│    Final Test    │' ));
  console.log( chalk.red( '└──────────────────┘' ));
  console.table( accuracyResult );
  console.log( `Parameters saved at: ${ chalk.yellow( savedPath ) }` );
  console.log( 'Done.' );
}

module.exports = {
  logEpoch,
  logFinalResult,
  getIndexOfBiggestNumberInArray,
  showImage,
  loadParameters,
  saveParameters
};
