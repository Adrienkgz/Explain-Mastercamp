import pandas as pd 

df = pd.read_csv("EFREI - LIPSTIP - 50k elements EPO balanced.csv")

print(df.keys())

# On sépare le dataset en 2 parties : 80% pour l'entrainement et 20% pour le test
train_df = df[:int(0.8*len(df))]
test_df = df[int(0.8*len(df)):]

train_df.to_csv("EFREI - LIPSTIP - 50k elements EPO balanced train.csv", index=False)
test_df.to_csv("EFREI - LIPSTIP - 50k elements EPO balanced test.csv", index=False)
