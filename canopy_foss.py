import os
import ogr
import gdal
import numpy
import json
import sklearn


def seperate_qq(naip_qq, num_areas):
  qq = naip_qq.GetLayer
  naip_fields = json.loads(qq.ExportToJson())
  
