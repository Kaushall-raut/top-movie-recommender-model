import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sea
from sklearn.metrics import confusion_matrix  
import matplotlib.pyplot as plot  

df =pd.read_csv('tmdb_movies_dataset.csv')

# print(df.head())

# print(df.info())

# rating, popularity ,  input features  and my target variable will movie names that will be recommended to the user

df['top_movie']=((df['rating']>=6.5)&(df['vote_count']>=400))

# print(df['top_movie'].head(20))/

# print(df.info())

# print(df.shape)  # dataset size 7 columns and 4000 data

# print(df['top_movie'].value_counts())


# data that i am going to use as in input feature 
x=df[['vote_count','popularity']]
y=df['top_movie']

x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.3)   #for training and testing


model=LogisticRegression()

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print("accuracy ",accuracy_score(y_test,y_predict))

sea.set_theme(style='white')

cm=confusion_matrix(y_test,y_predict)

sea.heatmap(cm,annot=True,fmt="d",cmap='Blues')


plot.xlabel("Predicted")
plot.ylabel("Actual")
plot.title("confusion matrix")
plot.show()