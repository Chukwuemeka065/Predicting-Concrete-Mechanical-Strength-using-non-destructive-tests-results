import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


TrainingsetFS=pd.read_csv("TrainingSet_70-30-FS.csv")
TestsetFS=pd.read_csv("TestingSet_70-30-FS.csv")

from autogluon.tabular import TabularPredictor

train_data = TrainingsetFS
test_data = TestsetFS

predictorFS = TabularPredictor(label='FS').fit(train_data=train_data)

TrainingsetTS=pd.read_csv("TrainingSet_70-30-TS.csv")
TestsetTS=pd.read_csv("TestingSet_70-30-TS.csv")

train_data = TrainingsetTS
test_data = TestsetTS

predictorTS = TabularPredictor(label='TS').fit(train_data=train_data)

TrainingsetCS=pd.read_csv("TrainingSet_70-30-CS.csv")
TestsetCS=pd.read_csv("TestingSet_70-30-CS.csv")

train_data = TrainingsetCS
test_data = TestsetCS

predictorCS = TabularPredictor(label='CS').fit(train_data=train_data)


def MechanicalStrength(UPV, ER):
    # Check if the inputs are valid (i.e., not empty and convertible to float)

    X = pd.DataFrame({'UPV': [UPV], 'ER': [ER]})
    predictionFS = round(float(predictorFS.predict(X, model='WeightedEnsemble_L2').iloc[0]), 4)
    predictionTS = round(float(predictorTS.predict(X, model='WeightedEnsemble_L2').iloc[0]), 4)
    predictionCS = round(float(predictorCS.predict(X, model='WeightedEnsemble_L2').iloc[0]), 4)
    return predictionFS, predictionTS, predictionCS
    
app = gr.Interface(fn=MechanicalStrength, 
                    inputs=[gr.components.Textbox(label="Ultrasonic Pulse Velocity (m/s)"),
                            gr.components.Textbox(label="Electrical Resistivity (ohm-cm)")],
                    outputs=[gr.components.Textbox(label="Flexural Strength (MPa)"),
                             gr.components.Textbox(label="Tensile Strength (MPa)"),
                             gr.components.Textbox(label="Compressive Strength (MPa)")],
                    description="Predicting Concrete Mechanical Strength with Ultrasonic Pulse Velocity and Electrical Resistivity data"
                  )

app.launch()
