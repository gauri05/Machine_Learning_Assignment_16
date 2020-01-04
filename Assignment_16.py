import pandas as pd
from sklearn.linear_model import LinearRegression

def Advertising_Predic():

    # Load data
    data = pd.read_csv('Advertising.csv')

    print("Size of data set",data.shape)

    X=data['TV'].values.reshape((-1,1))
    Y=data['sales'].values
    #Z=data['newspaper'].values

    reg=LinearRegression()

    reg=reg.fit(X,Y)

    y_pred =reg.predict(X)

    r2 =reg.score(X,Y)

    print(r2)

def main():
    print("Advertising Agency")
    Advertising_Predic()

if __name__=="__main__":
    main()