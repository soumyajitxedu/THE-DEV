from sklearn.linear_model import LogisticRegression
 
X =[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
Y =[1,0,0,1,0,1,1,0,1]
model = LogisticRegression()
model.fit(X,Y)
fuck = float(input("input "))
result = model.predict([[fuck]])[0]
if result == 1:
    print(f"based on hours{fuck}, you are chursi")
else:
    print("machodana")
