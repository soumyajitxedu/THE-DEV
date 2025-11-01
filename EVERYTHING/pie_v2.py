import matplotlib.pyplot as plt 
data = [12,34,13,83,22,38,9]
labs = ["porn","imgot","mango","ocata","palto","hapoi","piop"]
plt.pie(data, labels=labs,
        autopct="%1.1f%%",
        
        explode=[0,0,0,0,0.1,0.2,0.22],
        shadow=True,
        )
plt.title("pieeeeeeeee",size=23)
plt.show()