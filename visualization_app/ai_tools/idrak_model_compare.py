import pandas as pd
import os
import glob

import plotly.graph_objs as go

class IdrakModelCompare:
  def __init__(self,model_dir_path,classifier='*',dataset='*'):
    self.model_dir_path=model_dir_path
    self.dataset=dataset
    self.classifier=classifier
    self.model_report_paths=glob.glob(model_dir_path+f'{self.dataset}_{self.classifier}_report.json')

    self.models_names=[]
    self.models_acc=[]
    self.models_f1=[]
    self.models_recall=[]
    self.model_precision=[]
    self.report_df=[]
    self.ax=None
    self.process_model_reports()
  def process_model_reports(self):
    self.models_names=[os.path.basename(name)[:-12] for name in self.model_report_paths]
    self.models_acc=[pd.read_json(name)['Accuracy'][0] for name in self.model_report_paths]
    self.models_f1=[pd.read_json(name)['F1Score'][0] for name in self.model_report_paths]
    self.models_recall=[pd.read_json(name)['Recalll'][0] for name in self.model_report_paths]
    self.models_precision=[pd.read_json(name)['Precision'][0] for name in self.model_report_paths]
    self.report_df=pd.DataFrame([self.models_names,self.models_acc,self.models_f1,self.models_precision,self.models_recall]).T
    self.report_df.columns=['model_name','accuracy','f1_score','precision','recall']
    return self.report_df
  def visualize(self):
 
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=self.report_df['model_name'],
                        y=self.report_df["accuracy"],
                        name="Accuracy"))
    fig.add_trace(go.Bar(x=self.report_df['model_name'],
                        y=self.report_df["f1_score"],
                        name="F1 Score"))
    fig.add_trace(go.Bar(x=self.report_df['model_name'],
                        y=self.report_df["precision"],
                        name="Precision"))
    fig.add_trace(go.Bar(x=self.report_df['model_name'],
                        y=self.report_df["recall"],
                        name="Recall"))
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = self.report_df.index,
            ticktext = self.report_df.model_name)
    )
    
    return fig