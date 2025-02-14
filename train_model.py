import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import joblib as jb


url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic=pd.read_csv(url)
#Préparation des données
#gardons els colonnes utiles
data = titanic[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]] #6 features et target survived

#remplir les données manquantes : là, c'est l'age qu'on remplit par la moyenne
data["Age"].fillna(data["Age"].mean(), inplace=True)
#et Fare par la mediane
data["Fare"].fillna(data["Fare"].median(), inplace=True)

#encoder la colonne sex avec male=1 et female=0
data["Sex"] = LabelEncoder().fit_transform(data["Sex"])

#séparer les features X et la target Y
X=data.drop("Survived", axis=1).values
Y=data["Survived"].values

#normaliser les données
scaler=StandardScaler()
X=scaler.fit_transform(X)
jb.dump(scaler,"scaler.pkl")

#diviser les données en données de test et d'entrainement
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

#convertir les données en tensor pytorch
X_train=torch.tensor(X_train,dtype=torch.float32)
Y_train=torch.tensor(Y_train,dtype=torch.long)
X_test=torch.tensor(X_test,dtype=torch.float32)
Y_test=torch.tensor(Y_test,dtype=torch.long)

#creer dataloader
train_loader=DataLoader(TensorDataset(X_train,Y_train),batch_size=16,shuffle=True)
test_loader=DataLoader(TensorDataset(X_test,Y_test),batch_size=16,shuffle=False)

#modele du réseau neuronal

class TitanicNet(nn.Module):
    def __init__(self):
        super(TitanicNet,self).__init__()
        self.fc1=nn.Linear(6,16) #6 features, 16 neurones cachés
        self.fc2=nn.Linear(16,8) # 8 neurones cachés
        self.fc3=nn.Linear(8,2) #2 sorties 0 ou 1
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(0.2) # ADDED: Define dropout layer

    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


#initialsier le modele, la loss function et l'optimiseur

model=TitanicNet()
model.apply(init_weights) # MOVED: Apply weight initialization *before* training

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)

#entrainement du modele
num_epochs=50
for epoch in range(num_epochs):
    model.train()
    for inputs,labels in train_loader:
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")

#sauvegarde le modele
torch.save(model.state_dict(),"titanic_model.pth")
