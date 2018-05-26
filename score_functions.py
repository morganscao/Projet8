import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# Méthodes d'import export
import pickle
import os
from sklearn.externals import joblib


# Répertoires locaux
CT_DIR_DATA = 'C:/Users/Mscao/OneDrive - INTESCIA/Export/'
CT_DIR_SAVE = 'C:/Users/Mscao/OneDrive - INTESCIA/Export/save/'
# Répertoires sur le réseau
#CT_DIR_DATA = '//GESTIONSAGE1/Fichiers/Services S&D/S&D/6_ProjetsInterne/Morgan/export/'
#CT_DIR_SAVE = '//GESTIONSAGE1/Fichiers/Services S&D/S&D/6_ProjetsInterne/Morgan/save/'

# Méthode de suppression de colonne
def DropCol(p_df, p_col):
    if p_col in p_df.columns:
        p_df = p_df.drop([p_col], axis=1)
    return p_df

def save_obj(obj, name):
    fn = CT_DIR_SAVE + name + '.pkl'
    try:
        os.remove(fn)
    except OSError:
        pass
    with open(fn, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print(fn, 'saved')

def load_obj(name):
    with open(CT_DIR_SAVE + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def save_sklearn_obj(obj, name):
    fn = CT_DIR_SAVE + name + '.pkl'
    try:
        os.remove(fn)
    except OSError:
        pass
    joblib.dump(obj, fn)
    print(fn, 'saved')

def load_sklearn_obj(name):
    return joblib.load(CT_DIR_SAVE + name + '.pkl')
	
# Outliers
# Les entreprise qui ont subies une absorption se caractérisent soit par un procol = A
# soit par l'apparition d'un des évènements de la liste : 2620, 2621, 2720, 2725, 5700, 5500, 5501, 5502, 5503, 5510
def RemoveAbsorptions(p_df):
    if not 'lastTypeEven' in p_df.columns:
        print("Absorptions déjà enlevées")
        return p_df
    nb = p_df.shape[0]
    p_df['lastTypeEven'].fillna('', inplace=True)
    p_df['lastTypeEven2'].fillna('', inplace=True)
    p_df['lastTypeEvents'] = p_df['lastTypeEven'] +';'+ p_df['lastTypeEven2']
    p_df = DropCol(p_df, 'lastTypeEven')
    p_df = DropCol(p_df, 'lastTypeEven2')
    # On enlève les procol A
    p_df = p_df[(p_df['procol']!='A')]
    # On enlève les évènements de la liste
    p_df = p_df[~(p_df['lastTypeEvents'].str.contains('2620') | p_df['lastTypeEvents'].str.contains('2621') | 
          p_df['lastTypeEvents'].str.contains('2720') | p_df['lastTypeEvents'].str.contains('2725') | 
          p_df['lastTypeEvents'].str.contains('5700') | p_df['lastTypeEvents'].str.contains('5500') | 
          p_df['lastTypeEvents'].str.contains('5501') | p_df['lastTypeEvents'].str.contains('5502') | 
          p_df['lastTypeEvents'].str.contains('5503') | p_df['lastTypeEvents'].str.contains('5510'))]
    print(p_df.shape[0] - nb, "absorptions")
    p_df = DropCol(p_df, 'lastTypeEvents')
    return p_df
	
def fillBlanks(df):
    print('*** fillBlanks ***')
    # On remplace par ''
    for i in ['procol', 'procolMoins1', 'procolMoins2', 'procolMoins3', 'procolMoins4']:
        df[i] = df[i].fillna('')

    # On remplace par la valeur la plus courante
    #for i in ['ii_ORIGINE', 'ii_DAPET', 'ii_EXPLET', 'ii_APE_ENT', 'ii_TEFF_ENT', 'ii_ADR_DEP']:
    for i in df.columns:
        if df[i].dtype == object:
            if df[i].isnull().sum() > 0:
                df[i] = df[i].fillna(df[i].mode())
                print('\tNan de', i, 'remplacés par', df[i].mode()[0])

    # On remplace par la valeur moyenne
    for i in ['indiScoreMoins1', 'indiScoreMoins2', 'indiScoreMoins3', 'indiScoreMoins4']:
        if df[i].isnull().sum() > 0:
            mean = np.mean(df[i])
            df[i] = df[i].fillna(mean)
            print('\tNan de', i, 'remplacés par', mean)

    # On remplace le reste par 0
    for i in df.columns:
        if df[i].isnull().sum() > 0:
            df[i] = df[i].fillna(0)
            print('\tNan de', i, 'remplacés par 0')

    return df

# Transformation de la feature Date en age
def anneeToAge(value):
    if value > 1800:
        return dt.datetime.now().year - value
    return value

# Transformation du code NAF de niveau 5 en code NAF de niveau 1
def nafToNaf1(value):
    if len(value) < 2: return ''
    n = int(value[:2])
    if n<4: return 'A'
    if n<10: return 'B'
    if n<35: return 'C'
    if n<36: return 'D'
    if n<40: return 'E'
    if n<45: return 'F'
    if n<49: return 'G'
    if n<55: return 'H'
    if n<58: return 'I'
    if n<64: return 'J'
    if n<68: return 'K'
    if n<69: return 'L'
    if n<77: return 'M'
    if n<84: return 'N'
    if n<85: return 'O'
    if n<86: return 'P'
    if n<90: return 'Q'
    if n<94: return 'R'
    if n<97: return 'S'
    if n<99: return 'T'
    if n==99: return 'U'
    return ''

# Définition de la target : procol ou pas procol
def setTarget(value):
    return (len(value) == 0)

def showMissingValues(df):
    print('*** showMissingValues ***')
    tab_info = pd.DataFrame(df.dtypes).T.rename(index={0:'Type'})
    tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'Valeurs manquantes (nb)'}))
    tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0:'Valeurs manquantes (%)'}))
    display(tab_info)
    # Plus graphique
    x1=df.isnull().sum().values
    x2=df.notnull().sum().values
    index = np.arange(len(x1))
    fig, ax = plt.subplots(figsize=(16, 22))
    plt.barh(index, x1+x2, color='b')
    plt.barh(index, x1, color='r')
    ax.set_yticks(index)
    ax.set_yticklabels(df.columns.values.tolist())
    plt.show()

def complete_and_clean(dfPred, bBilan=False):
    print('*** complete_and_clean ***')
    dropcols = ['indiScoreDate', 'indiScoreDateMoins1', 'indiScoreDateMoins2', 'indiScoreDateMoins3'
            , 'indiScoreDateMoins4', 'ii_PROCOL']

    if not bBilan:
        print('Suppression des bilans financiers...')
        for col in dfPred.columns:
            if col.startswith('jb_'):
                # Pas de bilan pour l'instant (memory error)
                dfPred.drop([col], axis=1, inplace=True)

    # On force les features catégorielles
    dfPred ["ii_ACTIVNAT"] = dfPred ["ii_ACTIVNAT"].astype(np.str)
    dfPred ["ii_MODET"] = dfPred ["ii_MODET"].astype(np.str)
    dfPred ["ii_CJ"] = dfPred ["ii_CJ"].astype(np.str)
    dfPred ["ii_TCA"] = dfPred ["ii_TCA"].astype(np.str)
    dfPred ["ii_TCAEXP"] = dfPred ["ii_TCAEXP"].astype(np.str)
    
    dfPred['ii_AGE'] = dfPred['ii_DAPET'].apply(anneeToAge)
    dropcols.append('ii_DAPET')

    dfPred['ii_NAF1'] = dfPred['ii_APE_ENT'].apply(nafToNaf1)
    print(dfPred['ii_APE_ENT'].nunique(), 'codes NAF différents')
    print(dfPred['ii_NAF1'].nunique(), 'codes NAF de niveau 1')
    dropcols.append('ii_APE_ENT')

    # Table des stats
    print('Lecture des statistiques')
    dfStats = pd.read_csv(CT_DIR_DATA + 'stats_5_ans_PROD.csv', sep=";", na_values=r"\0", low_memory=False)

    # Fusion
    df = pd.merge(dfPred, dfStats)
    print('\n')
    print(len(set(df.siren)), 'SIREN au total')

    # Description avant nettoyage
    display(df.describe())
    display(df.select_dtypes(exclude=[np.number]).describe())

    # Avec une note il y a 12 mois
    print(-df[~(df['indiScoreMoins1']>0)].shape[0], "sans indiScore il y a 12 mois, ou à 0")
    df = df[(df['indiScoreMoins1']>0)]
    
    # Absorptions
    df = RemoveAbsorptions(df)
    print('Reste', df.shape[0], 'SIREN')

    # Suppression de colonnes qui vont bien
    for col in dropcols:
        df = DropCol(df, col)

    # Analyse des valeurs manquantes
    showMissingValues(df)

    plt.title("Proportion de valeurs absentes")
    # Remplissage des champs vides
    df = fillBlanks(df)
    
    # Cible : procol ou pas procol
    df['target'] = df['procol'].apply(setTarget)

    print('\n')
    print(df[df['target']==1].shape[0], 'SIREN actifs il y a 12 mois')
    print('\t', df[(df['target']==1) & (df['indiScoreMoins1']>6)].shape[0], 'TP (True Positifs)')
    print('\t', df[(df['target']==1) & (df['indiScoreMoins1']<=6)].shape[0], 'FN (False Negatifs)')
    print(df[df['target']==0].shape[0], 'SIREN en défaut')
    print('\t', df[(df['target']==0) & (df['indiScoreMoins1']>6)].shape[0], 'FP (False Positifs)')
    print('\t', df[(df['target']==0) & (df['indiScoreMoins1']<=6)].shape[0], 'TN (True Negatifs)')
    
    return df
	
# Charge les tables correspondant à une CJ donnée
def loadTableCJ(cj, bBilan=False):
    print('*** loadTableCJ ***')
    # Lecture
    n = CT_DIR_DATA + 'scores_predictions_PROD_CJ%s' % str(cj) + '.csv'
    print('reading', n, '...')
    dfPred = pd.read_csv(n, sep=";", na_values=r"\0", low_memory=False)
	# On complète
    return complete_and_clean(dfPred, bBilan)

# Charge toutes les tables
def loadTable(bBilan=False):
    print('*** loadTable ***')
    dfPred = pd.DataFrame()
    # Concaténation de toutes les données
    for i in range(1, 10):
        n = CT_DIR_DATA + 'scores_predictions_PROD_CJ%s' % str(i) + '.csv'
        print('reading', n, '...')
        f = pd.read_csv(n, sep=";", na_values=r"\0", low_memory=False)
        f['CJ'] = i
        dfPred = dfPred.append(f, ignore_index=True)
        print('\tshape ', f.shape)
    # Il faut reindexer si on a concaténé les fichiers 
    dfPred.index = range(len(dfPred.index))
    print(dfPred.shape[0], 'SIREN au total')
    # On complète
    df = complete_and_clean(dfPred, bBilan)
    # Quantité des différents éléments
    print('\n')
    display(pd.DataFrame([{'Entreprises': df['siren'].nunique(),    
               'NAF niveau 1': df['ii_NAF1'].nunique(),
               'Forme juridique': df['ii_CJ'].nunique(),
               'Situation juridique': df['procolMoins1'].nunique(),
              }], columns = ['Entreprises', 
                             'NAF niveau 1', 
                             'Forme juridique', 'Situation juridique'
                ], index = ['Quantité']))
    return df

def loadTable_old():
    print('*** loadTable ***')
    # Lecture
    dfPred = pd.DataFrame()
    lstSiren = [400, 500, 800, 1000]
    # Concaténation de toutes les données
    for i in lstSiren:
        n = CT_DIR_DATA + 'scores_predictions_PROD_%s' % str(i) + '.csv'
        print('reading', n, '...')
        f = pd.read_csv(n, sep=";", na_values=r"\0", low_memory=False)
        print('shape ', f.shape)
        dfPred = dfPred.append(f, ignore_index=True)
        # Il faut reindexer si on a concaténé les fichiers 
        dfPred.index = range(len(dfPred.index))
    print(dfPred.shape[0], 'SIREN')
    # On complète
    return complete_and_clean(dfPred)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepareData(p_df, p_CJ, p_dropcols, p_strat=None, bSave=True):
    # La target
    y = p_df['target']

    # On enlève la target des data bien sûr
    X = DropCol(p_df, 'target')
    # Ainsi que les colonnes non désirées
    for col in p_dropcols:
        # Sauf le score d'il y a 1 an, on s'en occupe plus loin
        if col == 'indiScoreMoins1': continue
        X = DropCol(X, col)

    scalingDFcols = []
    categDFcols = []
    for col in X.columns:
        if (X[col].dtype == object) or (X[col].dtype == np.str):
            categDFcols.append(col)
        else:
            scalingDFcols.append(col)
    print('Numérique :\n\t', scalingDFcols)
    print('Catégories :\n\t', categDFcols)

    categDF = X[categDFcols]
    scalingDF = X[scalingDFcols]
    
    # Binarisation en dummies pour garder la maitrise des noms des colonnes
    categDF_encoded = pd.get_dummies(categDF.astype(np.str))
    print('Après binarisation les catégories prennent', categDF_encoded.shape[1], 'dimensions.')

    # Concaténation
    x_final = pd.concat([scalingDF, categDF_encoded], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_final, y, test_size = 0.2, random_state=0, stratify=p_strat)
    
    # On comparera avec la prédiction actuelle, puis on l'enlève si demandé
    y_pred_actuelle = (x_test.indiScoreMoins1>6)
    if 'indiScoreMoins1' in p_dropcols:
        x_train = DropCol(x_train, 'indiScoreMoins1')
        x_test = DropCol(x_test, 'indiScoreMoins1')

    # Seules les xnum premières colonnes sont numériques
    xnum = scalingDF.shape[1]
    x_train_numerical = x_train.iloc[:, 0:xnum]
    x_test_numerical = x_test.iloc[:, 0:xnum]
    x_final_numerical = x_final.iloc[:, 0:xnum]

    # Création d'un scaler pour les valeurs numériques 
    scaler = StandardScaler()
    # Qu'on entraine avec le training set
    scaler.fit(x_train_numerical) 

    if bSave:
        # Sauvegarde
        save_obj(scalingDFcols, p_CJ+'_model_columnsScale'+str(x_final.shape))
        save_obj(categDFcols, p_CJ+'_model_columnsCateg'+str(x_final.shape))
        save_obj(x_final.columns, p_CJ+'_model_columns'+str(x_final.shape))
        save_sklearn_obj(scaler, p_CJ+'_model_scaler'+str(x_final.shape))
    
    x_train_numerical = scaler.transform(x_train_numerical)
    x_test_numerical = scaler.transform(x_test_numerical)
    x_final_numerical = scaler.transform(x_final_numerical)

    x_train = x_train.copy()
    x_test = x_test.copy()
    x_final = x_final.copy()
    x_train.loc[:, 0:xnum] = x_train_numerical
    x_test.loc[:, 0:xnum] = x_test_numerical
    x_final.loc[:, 0:xnum] = x_final_numerical

    print('x_train :', x_train.shape)
    return x_train, x_test, y_train, y_test, y_pred_actuelle, x_final


# Matrice de confusion
import itertools
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=2)
class_names = ['False', 'True']

# Affichage de matrice de confusion
def plot_confusion_matrix(cm, classes=['False', 'True'], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # Possibilité de normalisation
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')
    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

# Réduction dimensionnelle
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
def select_n_components(var_ratio, goal_var: float) -> int:
    total_variance = 0.0
    n_components = 0
    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break
    # Return the number of components
    print('Nouvelle dimension=',n_components)
    return n_components
def testPCA(X):
    # Standardize the feature matrix
    X = StandardScaler().fit_transform(X)
    # Create and run an TSVD with one less than number of features
    reduc = PCA(n_components=0.95, whiten=True)
    X_red = reduc.fit_transform(X)
    print('Original number of features:', X.shape[1])
    print('Nouvelle dimension:', X_red.shape[1])
    return reduc, X_red
def testTSVD(X):
    # Standardize the feature matrix
    X = StandardScaler().fit_transform(X)
    # Make sparse matrix
    X_sparse = csr_matrix(X)
    # Create and run an TSVD with one less than number of features
    reduc = TruncatedSVD(n_components=X_sparse.shape[1]-1)
    X_red = reduc.fit_transform(X)
    select_n_components(reduc.explained_variance_ratio_, 0.95)
    return reduc, X_red


def displayPCA(p_df, p_color):
    X_scaled = StandardScaler().fit_transform(p_df.fillna(0))
    
    pca = PCA(n_components=None)
    pca.fit(X_scaled)
    nbvar = len(pca.explained_variance_ratio_) +1

    fig, ax = plt.subplots(figsize=(16, 8))
    plt.bar(range(1,nbvar), pca.explained_variance_ratio_, alpha = 0.5, align = 'center', label = 'individual explained variance')
    plt.step(range(1,nbvar), np.cumsum(pca.explained_variance_ratio_), where = 'mid', label = 'cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc = 'best')
    plt.title('Variance cumulée', fontsize=18)
    plt.show()
    print("Deux composantes nous permettent d'expliquer %.2f pourcent de la variance" % (np.cumsum(pca.explained_variance_ratio_[:2])[1]*100))
    print('\n')
    
    # projeter X sur les composantes principales
    X_projected = pca.transform(X_scaled)
    # afficher chaque observation
    fig = plt.figure(figsize=(16, 10))
    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=p_color)
    plt.xlim([-5.5, 5.5])
    plt.ylim([-4, 4])
    plt.colorbar()
    plt.title('Projection sur les composantes principales', fontsize=18)
    plt.show()

    # S'il y a trop de feature on n'affichage pas ce dernier graphe qui sera illisible
    if nbvar > 15: return
    print('\n')
    pcs = pca.components_
    fig = plt.figure(figsize=(16, 10))
    for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
        # Afficher un segment de l'origine au point (x, y)
        plt.plot([0, x], [0, y], color='k')
        # Afficher le nom (data.columns[i]) de la performance
        plt.text(x, y, p_df.columns[i], fontsize='18')
    # Afficher une ligne horizontale y=0
    plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')
    # Afficher une ligne verticale x=0
    plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')
    plt.xlim([-0.7, 0.7])
    plt.ylim([-0.7, 0.7])
    plt.title('Contribution de chaque variable aux composantes principales', fontsize=18)
    plt.show()

    return