import glob
import os
import warnings
import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from pylab import rcParams
from tqdm.auto import tqdm

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

class IdrakModelResultVisualizer:
  def __init__(self,model_name='',history_filepath=None,invalid_prediction_filepath=None,report_file_path=None):
    self.model_name=model_name
    self.HISTORY_VALID=False
    if os.path.isfile(self.model_name+'_history.csv'):
      self.df_history=pd.read_csv(self.model_name+'_history.csv')
      self.HISTORY_VALID=True
      self.plot_history()
    self.df_invalid_predictions=pd.read_csv(self.model_name+'_invalid_predictions.csv')
    self.df_report=pd.read_json(self.model_name+'_report.json')
    self.plot_confusion_matrix()
    self.print_report()
  def plot_confusion_matrix(self):
    cnf_mat=self.df_report['ConfusionMatrix'].values[0]
    df_cm = pd.DataFrame(cnf_mat)
    plt.figure(figsize = (10,7))
    plt.title('Confusion Matrix')
    sns.heatmap(df_cm, annot=True, fmt='g')
    return df_cm
  def invalid_predictions(self):
    print(self.df_invalid_predictions)
    return self.df_invalid_predictions
  def print_report(self):
    print(self.df_report['ClassificationReport'][0])
    return self.df_report['ClassificationReport'][0]
  def plot_history(self):
    f1 = plt.figure(1)
    ax1=self.df_history['train_acc'].plot(title='Metrics Accuracy')
    ax1=self.df_history['valid_acc'].plot(title='Metrics Accuracy')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    f1.show()
    f2 = plt.figure(2)
    ax2=self.df_history['train_loss'].plot(title='Metrics Loss')
    ax2=self.df_history['valid_losss'].plot(title='Metrics Loss')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("losses")
    f2.show()

datadir= config.DATADIR
ai_model_dir='tinybert_report/'
model_name='dataset3_classifier_2'
model_path=datadir+ai_model_dir+model_name

irv=IdrakModelResultVisualizer(model_name=model_path)
