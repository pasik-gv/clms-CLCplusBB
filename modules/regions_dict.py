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

parks_info = {'Swietokrzyski National Park': ["Swietokrzyski National Park is a National Park in Swietokrzyskie Voivodeship in central Poland. The Swietokrzyskie Mountains are the oldest in Poland. Elevated in three different tectonic periods, they spread out in the Malopolska Upland, between Pilica and the Vistula river. Their outlines are gentle and their heights are small, the highest peak is Lysica at 612 meters."
"\n\n The history of efforts to protect this part of Poland dates back to the times before World War I. The first forest reserve was established in 1921, and expanded in subsequent years to eventually form the National Park in 1950. The area of today's Park is approximately 76 square kilometres, of which 72 km2 are forested. There are five strictly protected zones with a total area of over 17 km2."
"\n\n The park is famous for its trees, of which 674 are regarded as monuments of nature and as such are under protection. Among them is a 270-year-old European silver fir measuring 51 meters, considered the tallest conifer in Poland. The park's fauna is represented by more than 4000 species of invertebrates and 210 species of vertebrae (including 187 protected). Some of the large mammals found in the park include the deer (whose silhouette appears in Park's logo), wild boar, beaver and fox. Much less common, making occasional appearances are Moose and Nyctereutes. "
"Among many birds of prey nesting in the park are 4 species of owls: the Eurasian pygmy owl, Boreal owl, Long-eared owl and the rare and majestic Ural Owl."]}