import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

InputFile = "./diabetes.csv"

df = pd.read_csv(InputFile, header = 0)
features = list(df.columns[0:8])
y = df["Outcome"]
x = df[features]


clf = tree.DecisionTreeClassifier()
clf .fit(x, y)
print (clf.predict([[   0,0,0,0,0,0,0,0  ]]))


targetname = ['notdiabete', 'diabete']
featname = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, 
                            out_file=dot_data,
                            feature_names=featname,
                            class_names=targetname,
                            filled=True, rounded=True,
                            impurity=False)

graphs = pydot.graph_from_dot_data(dot_data.getvalue())

graphs[0].write_pdf("irisssde21.pdf")


