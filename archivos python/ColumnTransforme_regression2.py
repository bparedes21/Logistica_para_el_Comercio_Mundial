def funcion_ColumnTransforme(df):
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing

    label_encoder = preprocessing.LabelEncoder()

    df['Gender_Binary'] = label_encoder.fit_transform(df['Gender']) 
    df = pd.concat([df, pd.get_dummies(df['Mode_of_Shipment'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Warehouse_block'])], axis=1)
    df['Product_importance']=df['Product_importance'].replace("low", 0)
    df['Product_importance']=df['Product_importance'].replace("medium", 1)
    df['Product_importance']=df['Product_importance'].replace("high", 2)
    from sklearn.preprocessing import StandardScaler
    Standar =df[["Customer_care_calls","Cost_of_the_Product","Prior_purchases" ,"Discount_offered","Weight_in_gms"]]
    Standar_scaled_features = StandardScaler().fit_transform(Standar.values)    

    col=["Customer_care_calls","Cost_of_the_Product","Prior_purchases" ,"Discount_offered","Weight_in_gms"]
    df1=pd.DataFrame(Standar_scaled_features)
    #data_ML_car.loc[data_ML_car['carwidth']]=df[
    df1.columns=col
    df[col]=df1[col]
    col=["A","B","C","D","F","Road","Ship","Flight","Gender_Binary","Customer_rating","Customer_care_calls","Gender",'Warehouse_block','Mode_of_Shipment','Product_importance']
    df=df.drop(columns=col)
    return df
    #Cost_of_the_Product	Prior_purchases	Discount_offered	Weight_in_gms	Reached.on.Time_Y.N
def funcion_regression_log(df_train,df1_test):
    import pandas as pd
   
    X = df_train.iloc[:, [0,1,2,3]].values
    y =df_train.iloc[:, 4].values
    #divido los datos en test y entrenamiento
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

    from sklearn.linear_model import LogisticRegression 
    classifier = LogisticRegression(random_state = 10) 
    classifier.fit(X_train, y_train)

    X = df1_test.iloc[:, [0,1,2,3]].values

    res=classifier.predict(X)
    col=["pred"]
    df3=pd.DataFrame(res)
 
    df3.columns=col
    return df3


