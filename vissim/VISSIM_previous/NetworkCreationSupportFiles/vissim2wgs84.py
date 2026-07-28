from math import atan, exp, cos
from pyproj import Transformer

# variables from vissim net settings
xRefNet = 0.000
yRefNet = 0.000
xRefMap = -9485531.762
yRefMap = 4165877.412

# constant values
PI = 3.14159265358979
EarthRadius = 6378137
CorrectionFactorMercator = 1.0011202320000001

# deriving vissim local scal factor
LatitudeRefPointMap = ( 2 * atan( exp( CorrectionFactorMercator * yRefMap / EarthRadius ) ) - PI / 2 ) / ( PI / 180 )
LocalScaleFactor = 1 / cos( LatitudeRefPointMap * PI / 180 )

# vissim coordinates to be transformed
xVissim = -308.909
yVissim = -481.288

# calculating xy coordinates in Mercator
xMercator = ( xVissim - xRefNet ) * LocalScaleFactor + xRefMap
yMercator = ( yVissim - yRefNet ) * LocalScaleFactor + yRefMap

# transform Mercator coordinates to WGS84
merc2wgs84 = Transformer.from_crs('ESRI:53004', 'EPSG:4326')
wgs84 = merc2wgs84.transform(xMercator, yMercator)
print(wgs84)