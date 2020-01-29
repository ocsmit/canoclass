import os, sys
import ogr
from osgeo import gdal, ogr
import numpy
import json
import sklearn
from math import ceil

def seperate_qq(naip_qq, num_areas):
  qq = naip_qq.GetLayer
  naip_fields = json.loads(qq.ExportToJson())
  
  # Get a Layer's Extent
  inShapefile = "states.shp"
  inDriver = ogr.GetDriverByName("ESRI Shapefile")
  inDataSource = inDriver.Open(inShapefile, 0)
  inLayer = inDataSource.GetLayer()
  extent = inLayer.GetExtent()
  
  # Create a Polygon from the extent tuple
  ring = ogr.Geometry(ogr.wkbLinearRing)
  ring.AddPoint(extent[0],extent[2])
  ring.AddPoint(extent[1], extent[2])
  ring.AddPoint(extent[1], extent[3])
  ring.AddPoint(extent[0], extent[3])
  ring.AddPoint(extent[0],extent[2])
  poly = ogr.Geometry(ogr.wkbPolygon)
  poly.AddGeometry(ring)
  
  # total number of blocks / num_areas = 'i'
  # areas = math.round(i)
  # if not field process_area_id exists:
  #   create field
  # if field process_area_id exists:
  #   for j in naip_qq fields :
  #       every areas fields input j + 1
  
  #Fishnet gdal 
  def main(outputGridfn,xmin,xmax,ymin,ymax,gridHeight,gridWidth):

    # convert sys.argv to float
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridWidth = float(gridWidth)
    gridHeight = float(gridHeight)

    # get rows
    rows = ceil((ymax-ymin)/gridHeight)
    # get columns
    cols = ceil((xmax-xmin)/gridWidth)

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridHeight

    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputGridfn):
        os.remove(outputGridfn)
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn,geom_type=ogr.wkbPolygon )
    featureDefn = outLayer.GetLayerDefn()

    # create grid cells
    countcols = 0
    while countcols < cols:
        countcols += 1

        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom =ringYbottomOrigin
        countrows = 0

        while countrows < rows:
            countrows += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outLayer.CreateFeature(outFeature)
            outFeature = None

            # new envelope for next poly
            ringYtop = ringYtop - gridHeight
            ringYbottom = ringYbottom - gridHeight

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth

    # Save and close DataSources
    outDataSource = None
  
