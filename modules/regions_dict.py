'''
This file contains the dictionary of regions that can be selected in the main script.

The key of the dictionary is the name of the region, the value is a list with the following elements:
  - the coordinates of region, for the maps (latitude, longitude)
  - the default zoom level for the maps
  - the name of the region as it appears in the raster filename (in directory aoi_rasters/)
'''

regions_dict = {
    'Babiogorski National Park': [(49.59, 19.53), 12, 'Babiogorski_National_Park'],
    'Bialowieski National Park': [(52.76, 23.80), 11, 'Bialowieski_National_Park'],
    'Biebrzanski National Park': [(53.49, 22.96), 10, 'Biebrzanski_National_Park'],
    'Bieszczadzki National Park': [(49.10, 22.66), 11, 'Bieszczadzki_National_Park'],
    'Bory Tucholskie National Park': [(53.81, 17.56), 11, 'Bory_Tucholskie_National_Park'],
    'Drawienski National Park': [(53.10, 15.90), 11, 'Drawienski_National_Park'],
    'Gorczanski National Park': [(49.55, 20.13), 12, 'Gorczanski_National_Park'],
    'Gory Stolowe National Park': [(50.46, 16.35), 12, 'Gory_Stolowe_National_Park'],
    'Kampinoski National Park': [(52.32, 20.57), 11, 'Kampinoski_National_Park'],
    'Karkonoski National Park': [(50.79, 15.62), 11, 'Karkonoski_National_Park'],
    'Magurski National Park': [(49.52, 21.44), 11, 'Magurski_National_Park'],
    'Narwianski National Park': [(53.05, 22.87), 11, 'Narwianski_National_Park'],
    'Ojcowski National Park': [(50.20, 19.82), 12, 'Ojcowski_National_Park'],
    'Pieninski National Park': [(49.41, 20.37), 12, 'Pieninski_National_Park'],
    'Poleski National Park': [(51.42, 23.19), 11, 'Poleski_National_Park'],
    'Roztoczanski National Park': [(50.59, 23.02), 11, 'Roztoczanski_National_Park'],
    'Slowinski National Park': [(54.69, 17.30), 11, 'Slowinski_National_Park'],
    'Swietokrzyski National Park': [(50.89, 20.94), 11, 'Swietokrzyski_National_Park'],
    'Tatrzanski National Park': [(49.25, 19.93), 11, 'Tatrzanski_National_Park'],
    'Ujscie Warty National Park': [(52.60, 14.78), 11, 'Ujscie_Warty_National_Park'],
    'Wielkopolski National Park': [(52.27, 16.76), 11, 'Wielkopolski_National_Park'],
    'Wigierski National Park': [(54.03, 23.10), 11, 'Wigierski_National_Park'],
    'Wolinski National Park': [(53.91, 14.47), 11, 'Wolinski_National_Park']
}
