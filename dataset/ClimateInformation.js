////////////////////////////////////////////////////////////////////////////////////////////
// This code exports Climate features for the LUCAS samples in Germany
// Data: https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE
// Author: Nafiseh Kakhani, University of Tuebingen
// Date: 13/3/2023
// Project: Multi-modal deep learning for SOC estimation 
////////////////////////////////////////////////////////////////////////////////////////////

// Import samples
var region = ee.FeatureCollection("..."); //Add your feature collection here
var Climatefeature = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').select('aet');
// Desired bands:[aet, def, pdsi, pet, pr, ro, soil, srad, swe, tmmn, tmmx, vap, vpd, vs]

var bandName = ee.Image(Climatefeature.first()).bandNames().get(0);

var startDate = ee.Date('2015-01-01'); // set analysis start time
var endDate = ee.Date('2015-12-31'); // set analysis end time

// calculate the number of months to process
var nMonths = ee.Number(endDate.difference(startDate,'month')).round();
// the number of months starts from 0, so we need to reduce nMonths by one.
var nMonths = ee.Number(nMonths).subtract(1)

// get a list of time strings to pass into a dictionary later on
var monList = ee.List(ee.List.sequence(0,nMonths).map(function (n){
  return startDate.advance(n,'month').format('YYYMMdd');
}))

print(monList)

function map_climate(feature){
    // map over each month
    var timeSeries = ee.List.sequence(0,nMonths).map(function (n){
    // calculate the offset from startDate
    var ini = startDate.advance(n,'month');
    // advance just one month
    var end = ini.advance(1,'month');
    // filter and reduce
    var data = Climatefeature.filterDate(ini,end).mean().reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: feature.geometry(),
    });
    
    // get the value and check that it has data
    var val = ee.Number(data.get(bandName));
    val = ee.Number(ee.Algorithms.If(val,val,-999));
    return val;
  });
  
  // create new dictionary with date strings and values
  var timeDict = ee.Dictionary.fromLists(monList,ee.List(timeSeries));
  // return feature with a timeseries property and results
  return feature.set(timeDict).setGeometry(null); //set the geometry to null to speed up the process
}

//description of files
var FILE_PREFIX = 'Climate_' + 'aet'
var FOLDER = 'LUCAS_de';
var desc_tr = FILE_PREFIX + '_' 

//export 
Export.table.toDrive({
    collection: region.map(map_climate),
    description: desc_tr,
    folder: FOLDER,
    fileFormat: 'CSV',
});
