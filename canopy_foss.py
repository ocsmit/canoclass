import os
import ogr
import gdal
import numpy
import json
import sklearn


def seperate_qq(naip_qq, num_areas):
  qq = naip_qq.GetLayer
  naip_fields = json.loads(qq.ExportToJson())
  
  # total number of blocks / num_areas = 'i'
  # if not field process_area_id exists:
  #   create field
  # if field process_area_id exists:
  #   for j in naip_qq fields :
  #       every i fields input j + 1
  
  
