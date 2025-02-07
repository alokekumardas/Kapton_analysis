nominal_sensorlength = 94.183
maximal_sensorlength = nominal_sensorlength+0.1

minimal_clearance_air = 0.5
nominal_clearance_L = 0.7

nominal_kaptonlength = 95.683
minimal_kaptonlength = nominal_kaptonlength-0.1 #steht nominal 750mu über. man will auf gar keinen fall unter einen Überstand von 500mu kommen.
nominal_kaptonwidth = 9.8
minimal_kaptonwidth = nominal_kaptonwidth-0.2 #-0.2 oder 0.25 ?
nominal_KaptonEdgeToSensorEdge = 0.15
maximal_kaptonlength = nominal_kaptonlength+0.15 #0.15 wenn für barrel. sonst geht auch größer (+0.3mm). Problem: da ist auf der Halterung nicht so viel Luft!
maximal_kaptonwidth = nominal_kaptonwidth+0.5 # hier hat man mehr Toleranz, da keine "Halterung" im Weg

nominal_stumpkaptonwidth = 10.1
nominal_stumpkaptonlength = 17.26
nominal_stumpbridgewidth = 8.0
nominal_stumpbridge_distance_shortedgekapton = 1.1
nominal_stumpbridge_distance_longedgekapton = 1.05
nominal_stumpbridge_lengthonsensor=15.4615
minimal_stumpkapton_width = nominal_stumpkaptonwidth-0.25 #-0.2 oder 0.25 ? Nomineller Überstand je Seite 1.05mm. benötigt 500µm je Seite. Allerdins stump bridge nicht immer perfekt gerade...
maximal_stumpkapton_width = nominal_stumpkaptonwidth+0.5 
minimal_stumpkapton_length = nominal_stumpkaptonlength-0.5 # 18.7.24: ursprünglich -0.1 wie bei den langen Streifen, aber das ist quatsch da ca 7mm länger als die Brücke
maximal_stumpkapton_length = nominal_stumpkaptonlength+0.5 #18.7.24: vorher +0.15 wie bei den langen Streifen aber das ist Quatsch, zu lang ist eigentlich ziemlich egal

nominal_AlCFwidth = 8.0
nominal_kapton_thickness = 25E-3

constant_unit = 'mm'

"""pre 27.03.24 
defaultFilter_R = [235, 255] 
defaultFilter_G = [185, 210] 
defaultFilter_B = [30, 65]
"""
defaultFilter_R = [225, 255] 
defaultFilter_G = [182, 255] 
defaultFilter_B = [0, 95] #0

"""
defaultFilter_R = [240, 252] 
defaultFilter_G = [200, 240] 
defaultFilter_B = [40, 75]

defaultFilter_R = [240, 255] 
defaultFilter_G = [200, 245] 
defaultFilter_B = [40, 80]
"""
old_templateFilter_R = [180, 255]
old_templateFilter_G = [0, 140]
old_templateFilter_B = [0, 140]

''' #settings as currentlyin in the git, but probably not used for the analysis...
yTemplateFilter_R = [0, 60]
yTemplateFilter_G = [0, 60]
yTemplateFilter_B = [0, 60]

xTemplateFilter_R = [0, 80]
xTemplateFilter_G = [0, 80]
xTemplateFilter_B = [0, 80]
'''
yTemplateFilter_R = [0, 58]
yTemplateFilter_G = [0, 58]
yTemplateFilter_B = [0, 58]

xTemplateFilter_R = [0, 78]
xTemplateFilter_G = [0, 78]
xTemplateFilter_B = [0, 78]


templateDimensions1 = [95.6265, 9.784]
#templateDimensions2 = [95.6405, 9.774] #not used
templateDimensions3 = [95.688, 9.770]
templateDimensions4 = [95.657, 9.784]
templateDimensions5 = [95.659, 9.784]


plottingScale = 1
default_DPI = 1200
#correctionfactor = 0.007