project_path = 'F:/canopy_dev/'



data_path = '%s/Data' % project_path

phyregs_layer = '%s/Physiographic_Districts_GA.shp' % data_path


naipqq_layer = '%s/naip_ga_2009_1m_m4b.shp' % data_path

naipqq_phyregs_field = 'phyregs'


naip_path = '%s/NAIP 2009/ga' % project_path

# Well-Known IDs (WKIDs) are numeric identifiers for coordinate systems
# administered by Esri.  This variable specifies the target spatial reference
# for output files. For more information about WKIDs, please refer to the
# following sources:
#   https://developers.arcgis.com/rest/services-reference/projected-coordinate-systems.htm
#   https://pro.arcgis.com/en/pro-app/arcpy/classes/pdf/projected_coordinate_systems.pdf
#   http://resources.esri.com/help/9.3/arcgisserver/apis/rest/pcs.html
# WKID 102039 is USA Contiguous Albers Equal Area Conic USGS version.
spatref_wkid = 102039

analysis_path_format = '%s/%%d Analysis' % project_path

analysis_year = 2009


analysis_path = analysis_path_format % analysis_year

process_area_degree_size = 30

process_area_fishnet = '%s/fishnet.shp' % analysis_path

process_area_naipqq = '%s/process_area_naipqq.shp' % analysis_path

snaprast_path = '%s/rm_3408504_nw_16_1_20090824.tif' % data_path

# This folder will contain all result files.
results_path = '%s/Results' % analysis_path

index_out_path = '%s/%%s' % analysis_path
