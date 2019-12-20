
# import pandas as pd
# from sklearn import tree


# InputFile = "/root/Desktop/pythonkali/diabetes.csv"

# df = pd.read_csv(InputFile, header = 0)
# features = list(df.columns[0:8])
# y = df["Outcome"]
# x = df[features]

# clf = tree.DecisionTreeClassifier()
# clf.fit(x,y)


# print(clf.predict([[5,109,75,26,0,36,0.546,60]]))


# # #viz code 
# from sklearn.externals.six import StringIO
# import pydot
# dot_data = StringIO()
# tree.export_graphviz(clf,
#                             out_file=dot_data,
#                             feature_names=x,
#                             class_names=y,
#                             filled=True, rounded=True,
#                             impurity=False)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("diabetesss.pdf")


import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
test_idx = [0,50,100]

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)



from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, 
                            out_file=dot_data,
                            feature_names=iris.feature_names,
                            class_names=iris.target_names,
                            filled=True, rounded=True,
                            impurity=False)

graphs = pydot.graph_from_dot_data(dot_data.getvalue())
# print (graphs)
graphs[0].write_pdf("iris.pdf")