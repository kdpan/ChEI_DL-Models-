import pandas as pd
from tqdm import tqdm

def get_ChE_BBB_results(saved_results):
    ache = pd.read_csv('AchE-TCMSP-Screen-LMs.csv')
    bche = pd.read_csv('BchE-TCMSP-Screen-LMs.csv')
    bbb = pd.read_csv('tcmsp-bbb-prediction.csv')

    smi2bbb = {}
    for i,row in tqdm(bbb.iterrows(),total=bbb.shape[0]):
        smi2bbb[row['smi']] = round(row['BBBP score'],4)


    ID2P06276 = {}
    ID2Q03311 = {}
    ID2P81908 = {}

    for i,row in tqdm(bche.iterrows(),total=bche.shape[0]):
        ID2P06276[row['TCMSP ID']] = row['P06276 Score']
        ID2Q03311[row['TCMSP ID']] = row['Q03311 Score']
        ID2P81908[row['TCMSP ID']] = row['P81908 Score']

    results = pd.DataFrame(columns=['TCMSP ID','SMILES','BBBP score','P22303 Score','P23795 Score','P21836 Score','P06276 Score','Q03311 Score','P81908 Score'])
    results.to_csv(saved_results,mode='w',index=None)
    n = 0
    for i,row in tqdm(ache.iterrows(),total=ache.shape[0]):
        smi = row['smi']
        id_ = row['TCMSP ID']
        results.loc[n] = [id_,smi,smi2bbb.get(smi),row['P22303 Score'],row['P23795 Score'],row['P21836 Score'],ID2P06276.get(id_),ID2Q03311.get(id_),ID2P81908.get(id_)]
        n+=1
        results.to_csv(saved_results,mode='a',index=None,header=None)
        results = pd.DataFrame(columns=['TCMSP ID','SMILES','BBBP score','P22303 Score','P23795 Score','P21836 Score','P06276 Score','Q03311 Score','P81908 Score'])

get_ChE_BBB_results('TCMSP ChE Prediction.csv')