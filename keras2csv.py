import utils.tftools as tt
# tested models
    # 1. keras pretrianed models: 
        # 'DenseNet121',  'DenseNet169',  'DenseNet201',
        # 'InceptionResNetV2',  'InceptionV3',
        # 'MobileNet',  'MobileNetV2',
        # 'NASNetLarge', 'NASNetMobile',
        # 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 'ResNet50V2',
        # 'VGG16',  'VGG19',
        # 'Xception',
    # 2 Reomendeation: din
    # 3 EfficientNet: EfficientNetB0 ~ EfficientNetB7
    # 4 NLP: bert

# model to be loaded
nnname = 'ResNet50V2'

(model,isconv) = tt.GetModel(nnname) 

# Producing Parameter table of given Model 
paralist = tt.ListGen(model,isconv) 
    
# exproting tables to //outputs//tf
tt.tableExport(paralist,nnname)