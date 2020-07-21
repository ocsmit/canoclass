# ==============================================================================
# Title: config.py
# Author: Owen Smith, University of North Georgia
# Canopy data creation config:
# -----------------------------
#       * All proccess functions rely on the configurations set within this
#         file to run.
#
# ==============================================================================

# Output projection
proj = 'EPSG:5070'

# Folder where process will output results and read data from
workspace = '/mnt/Research/GFC_FOSS'

# Folder that contains all NAIP
naip_dir = '/mnt/Research/GFC/GFC 2015/2015 NAIP/ga'

# Folder where all regions will be output
results = '%s/Results' % workspace

# Folder within region folders that will contain final outputs after
# classification
class_directory = '%s/%%s/Outputs' % workspace

# Folder where all refernce and training data is stored.
data = '%s/Data' % workspace

# Physiographic districts shapefile
proc_region = '%s/Physiographic_Districts_GA.shp' % data

# Field with proccesing region ID's
procid_field = "PHYSIO_ID"

# Original NAIP QQ shapefile to use for clipping
clip_naip = '%s/ga_naip_clip.shp' % data

# Joined NAIP QQ tile with process region ID's to query filenames
naipqq_shp = '%s/ga_naip_re_test.shp' % data

# Rasterized training data
training_raster = '%s/training_raster_2.tif' % data

# ARVI raster which training data applies to
training_fit_raster = '%s/arvi_m_3408326_ne_17_1_20150915.tif' % data

