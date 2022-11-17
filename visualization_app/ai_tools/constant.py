DATADIR='data/model_metrics'
classifier_1_labelmaps={0:'answering_machine',1:'dnc'}
classifier_2_labelmaps={0:'answering_machine',1:'busy',2:'dnc',3:'greetback',4:'greeting',5:'other',6:'sorry_greeting',7:'spanish'}
classifier_3_labelmaps={0:'bot',1:'busy',2:'dnc',3:'negative',4:'not_intrested',5:'other',6:'positive',7:'spanish'}
classifier_4_labelmaps={0:'dnc',1:'negative',2:'not_intrested',3:'other',4:'positive'}
classifier_5_labelmaps={0:'dnc',1:'negative',2:'not_intrested',3:'other',4:'positive'}
classifiers_meta={'classifier_1':{'labels': classifier_1_labelmaps,'NUM_CLASSES':2},
                  'classifier_2':{'labels': classifier_2_labelmaps,'NUM_CLASSES':8},
                  'classifier_3':{'labels': classifier_3_labelmaps,'NUM_CLASSES':8},
                  'classifier_4':{'labels': classifier_4_labelmaps,'NUM_CLASSES':5},
                  'classifier_5':{'labels': classifier_5_labelmaps,'NUM_CLASSES':5}
                  }
dataset_names={'All Datasets':'*','Human Label':'dataset1','Current Transcripter':'dataset2','Beam':'dataset3','Greedy':'dataset4'}
classifier_names={'All Classifier':'*','Hello':'classifier_1','Intro':'classifier_2','Pitch':'classifier_3','Yes/No with Age Data':'classifier_4','Yes/No without Age Data':'classifier_5'}