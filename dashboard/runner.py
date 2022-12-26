import argparse
from termcolor import colored
from pyfiglet import figlet_format
import os

print((colored(figlet_format("Welcome to Idrak Bert Trainer"), color="green")))

from ai_tools.trainer_logistics import IdrakTinyBertClassifier 
# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data", help = "Name of Folder of Your Dataset \
                    inside contents/data/")

parser.add_argument("-m","--model", help = "Name of Intended Model; If dir  \
                    not exists it will create dir in models/. Just Given Any\
                    Name to model")

parser.add_argument("-e","--epochs", help = "Number of epochs to training  \
                      Bert Model. By Default it will be 5",default=5,type=int)

parser.add_argument("-l","--learning_rate", help = "Learning Rate (lr) to train \
                        Bert Model. By Default it will be 2e-4 ",default=2e-4,
                        type=float)

args = parser.parse_args()

 
if args.data:
    data_name=args.data
if args.model:
    model_path=args.model
    
if args.epochs:
    n_epochs=int(args.epochs)
else:
    n_epochs=5

if args.learning_rate:
    lr=float(args.learning_rate)
else:
    lr=2e-4


try:
    
    data_dir='contents/data/'
    report_dir='tinybert_results1/'
    dataset=f'{data_name}/'
    datapath=data_dir+dataset+dataset[:-1]
    model_name=model_path
    checkpoint_name=model_name
    model_name=report_dir+model_name
    
    runner_ready=True
    
except Exception as e:
    print(f'Error in arge parsing {e}')
    runner_ready=False
    


if runner_ready:
    print(datapath)
    itbc=IdrakTinyBertClassifier(datapath,num_epochs=n_epochs,learning_rate=lr,model_name=model_name,checkpoint_name=checkpoint_name)

