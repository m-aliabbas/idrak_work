from ai_tools.trainer_logistics import IdrakTinyBertClassifier

data_dir='contents/data/'
report_dir='tinybert_results1/'
dataset='multi_class/'
datapath=data_dir+dataset+dataset[:-1]
model_name='multiclass_test21'
checkpoint_name=model_name
model_name=report_dir+model_name

itbc=IdrakTinyBertClassifier(datapath,num_epochs=5,learning_rate=2e-4,model_name=model_name,checkpoint_name=checkpoint_name)

