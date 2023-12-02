import torch
from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer

from rdkit import Chem
from rdkit.Chem import MolFromSmiles

import re
import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import requests
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index


def embedding_model_init(device,model_name="Rostlab/prot_t5_xl_uniref50"):
    #@param {type:"string"}["Rostlab/prot_t5_xl_uniref50", "Rostlab/prot_t5_xl_bfd", "Rostlab/prot_t5_xxl_uniref50", "Rostlab/prot_t5_xxl_bfd", "Rostlab/prot_bert_bfd", "Rostlab/prot_bert", "Rostlab/prot_xlnet", "Rostlab/prot_albert"]
    print(model_name)
    if "t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False )
        model = T5EncoderModel.from_pretrained(model_name)
        shift_left = 0
        shift_right = -1
    elif "albert" in model_name:
        tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False )
        model = AlbertModel.from_pretrained(model_name)
        shift_left = 1
        shift_right = -1
    elif "bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False )
        model = BertModel.from_pretrained(model_name)
        shift_left = 1
        shift_right = -1
    elif "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False )
        model = XLNetModel.from_pretrained(model_name)
        shift_left = 0
        shift_right = -2
    else:
        print("Unkown model name")
    
    model = model.to(device)
    model = model.eval()
    # if torch.cuda.is_available():
    #     model = model.half() 
    return tokenizer,model,shift_left,shift_right

def seq_embedding(seq2path,model_name,max_seq_len):
    # use protTrans pretrain model for protein seqs Embedding
    pad_tokens = "0"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer,model,shift_left,shift_right = embedding_model_init(device,model_name)

    print("Embedding seqs ...")  
    for seq in tqdm(seq2path.keys()):
        s = seq
        with torch.no_grad():
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
                seq_mask = torch.ones(max_seq_len)
            else:
                seq_mask = torch.zeros(max_seq_len)
                seq_mask[:len(seq)] = 1
                pad_nums = max_seq_len-len(seq)
                seq += pad_tokens*pad_nums
            
            ids = tokenizer.batch_encode_plus([list(seq)], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
            embedding = model(input_ids=ids['input_ids'].to(device))[0]
            embed_seq = embedding[0].detach()[shift_left:shift_right]
            save_path = seq2path[s]
            torch.save((embed_seq,seq_mask),save_path)   

def process_seqs_from_GraphDTA():
    max_seq_len = 1024
    embedding_model_name = "Rostlab/prot_albert"
    #@param {type:"string"}["Rostlab/prot_t5_xl_uniref50", "Rostlab/prot_t5_xl_bfd", "Rostlab/prot_t5_xxl_uniref50", "Rostlab/prot_t5_xxl_bfd", "Rostlab/prot_bert_bfd", "Rostlab/prot_bert", "Rostlab/prot_xlnet", "Rostlab/prot_albert"]

    seqs_iso = []
    for dt_name in ['kiba','davis']:
        opts = ['train','test']
        for opt in opts:
            df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
            seqs_iso += list( df['target_sequence'] )
    seqs_iso = set(seqs_iso)
    if not os.path.exists("data/seq_embs"):
        os.makedirs("data/seq_embs")
    
    embed_dir = os.path.join('data/seq_embs',embedding_model_name.split('/')[-1])
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
    seq2path = {v:os.path.join(embed_dir,'{}.pt'.format(i)) for i,v in enumerate(seqs_iso)} 
    seq2path_filepath = os.path.join('data','seq2path_{}.pickle'.format(embedding_model_name.split('/')[-1]))
    
    with open(seq2path_filepath,'wb') as handle:
        pickle.dump(seq2path,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open(seq2path_filepath, 'rb') as handle:
        seq2path = pickle.load(handle)
    seq_embedding(seq2path,embedding_model_name,max_seq_len)

def process_smiles_from_GraphDTA():
    compound_iso_smiles = []
    for dt_name in ['kiba','davis']:
        opts = ['train','test']
        for opt in opts:
            df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
            compound_iso_smiles += list( df['compound_iso_smiles'] )
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)  
        smile_graph[smile] = g
    with open('data/smile_graph.pickle', 'wb') as handle:
        pickle.dump(smile_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)



def process_uniprot_seqs_from_GraphDTA(src):
    max_seq_len = 1024
    embedding_model_name = "Rostlab/prot_t5_xl_uniref50"
    #@param {type:"string"}["Rostlab/prot_t5_xl_uniref50", "Rostlab/prot_t5_xl_bfd", "Rostlab/prot_t5_xxl_uniref50", "Rostlab/prot_t5_xxl_bfd", "Rostlab/prot_bert_bfd", "Rostlab/prot_bert", "Rostlab/prot_xlnet", "Rostlab/prot_albert"]

    prot_info = pd.read_csv(src, sep='\t')
    if not os.path.exists("data/seq_embs"):
        os.makedirs("data/seq_embs")
    embed_dir = os.path.join('data/seq_embs',embedding_model_name.split('/')[-1])
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)

    seq2path = {}
    for i,row in prot_info.iterrows():
        seq = row['Sequence']
        prot_id = row['Entry']    
        seq2path[seq] = os.path.join(embed_dir,prot_id+'.pt')

    
    seq2path_filepath = os.path.join('data','seq2path_{}.pickle'.format(embedding_model_name.split('/')[-1]))
    
    with open(seq2path_filepath,'wb') as handle:
        pickle.dump(seq2path,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open(seq2path_filepath, 'rb') as handle:
        seq2path = pickle.load(handle)
    seq_embedding(seq2path,embedding_model_name,max_seq_len)

def process_tcmsp_smile_from_GraphDTA(src,dst):
    data = pd.read_csv(src)
    smi = list(set(data['smi'].to_list()))
    smile_graph = {}
    for s in tqdm(smi):
        g = smile_to_graph(s)  
        smile_graph[s] = g
    with open(dst, 'wb') as handle:
        pickle.dump(smile_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_sdf2graph_from_GraphDTA(src,dst):
    data = Chem.SDMolSupplier(src)
    smile_graph = {}

    for c in tqdm(data):
        try:
            smi = Chem.MolToSmiles(c)
        except:
            continue
        g = smile_to_graph(smi)  
        smile_graph[smi] = g
    with open(dst, 'wb') as handle:
        pickle.dump(smile_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def dataset_split(src):
    data = pd.read_csv(src)
    train,test = train_test_split(data,test_size=0.2)
    valid,test = train_test_split(test,test_size=0.5)
    train.to_csv('data/che_train.csv',index=None)
    valid.to_csv('data/che_valid.csv',index=None)
    test.to_csv('data/che_test.csv',index=None)

# dataset_split('abche_dataset.csv')

# process_tcmsp_smile_from_GraphDTA('screen_database/tcmsp.csv','screen_database/tcmsp.graph')
# process_sdf2graph_from_GraphDTA('screen_database/L6030.sdf','screen_database/L6030.graph')
# process_seqs_from_GraphDTA()
# process_smiles_from_GraphDTA()
# process_uniprot_seqs_from_GraphDTA('prot_info.tsv')

def screen_data_process(cp,targets):
    if cp.split('.')[-1] == 'csv':
        saved_file = 'prediction/data/tcmsp-bche.csv'
        cp_df = pd.read_csv(cp)
        data = pd.DataFrame(columns=['mol_id','prot_id','smi','seq'])
        data.to_csv(saved_file,mode='w',index=None)
        n=0
        for prot_id in targets.keys():
            seq = targets.get(prot_id)
            for i,row in tqdm(cp_df.iterrows(),total=cp_df.shape[0]):
                smi = row['smi']
                id_ = row['id']
                data.loc[n] = [id_,prot_id,smi,seq]
                data.to_csv(saved_file,mode='a',index=None,header=None)
                data = pd.DataFrame(columns=['mol_id','prot_id','smi','seq'])
                n+=1
    elif cp == "screen_database/drugbank_open_structures.sdf":
        cp = Chem.SDMolSupplier(cp)
        saved_file = 'prediction/data/drugbank-bche.csv'
        data = pd.DataFrame(columns=['mol_id','prot_id','smi','seq'])
        data.to_csv(saved_file,mode='w',index=None)
        n=0
        u = 0
        for prot_id in targets.keys():
            seq = targets.get(prot_id)
            u+=1
            for c in tqdm(cp,desc='{}/{}'.format(u,len(targets.keys()))):
                try:
                    smi = Chem.MolToSmiles(c)
                    id_ = c.GetProp('DRUGBANK_ID')
                except:
                    continue
                
                
                data.loc[n] = [id_,prot_id,smi,seq]
                data.to_csv(saved_file,mode='a',index=None,header=None)
                data = pd.DataFrame(columns=['mol_id','prot_id','smi','seq'])
                n+=1
    elif cp == "screen_database/L6030.sdf":
        cp = Chem.SDMolSupplier(cp)
        saved_file = 'prediction/data/L6030-ache.csv'
        data = pd.DataFrame(columns=['mol_id','prot_id','smi','seq'])
        data.to_csv(saved_file,mode='w',index=None)
        n=0
        u = 0
        for prot_id in targets.keys():
            seq = targets.get(prot_id)
            u+=1
            for c in tqdm(cp,desc='{}/{}'.format(u,len(targets.keys()))):
                try:
                    smi = Chem.MolToSmiles(c)
                    id_ = c.GetProp('IDNUMBER')
                except:
                    continue
                
                
                data.loc[n] = [id_,prot_id,smi,seq]
                data.to_csv(saved_file,mode='a',index=None,header=None)
                data = pd.DataFrame(columns=['mol_id','prot_id','smi','seq'])
                n+=1

BchE_trained_Prot = {'P06276':'MHSKVTIICIRFLFWFLLLCMLIGKSHTEDDIIIATKNGKVRGMNLTVFGGTVTAFLGIPYAQPPLGRLRFKKPQSLTKWSDIWNATKYANSCCQNIDQSFPGFHGSEMWNPNTDLSEDCLYLNVWIPAPKPKNATVLIWIYGGGFQTGTSSLHVYDGKFLARVERVIVVSMNYRVGALGFLALPGNPEAPGNMGLFDQQLALQWVQKNIAAFGGNPKSVTLFGESAGAASVSLHLLSPGSHSLFTRAILQSGSFNAPWAVTSLYEARNRTLNLAKLTGCSRENETEIIKCLRNKDPQEILLNEAFVVPYGTPLSVNFGPTVDGDFLTDMPDILLELGQFKKTQILVGVNKDEGTAFLVYGAPGFSKDNNSIITRKEFQEGLKIFFPGVSEFGKESILFHYTDWVDDQRPENYREALGDVVGDYNFICPALEFTKKFSEWGNNAFFYYFEHRSSKLPWPEWMGVMHGYEIEFVFGLPLERRDNYTKAEEILSRSIVKRWANFAKYGNPNETQNNSTSWPVFKSTEQKYLTLNTESTRIMTKLRAQQCRFWTSFFPKVLEMTGNIDEAEWEWKAGFHRWNNYMMDWKNQFNDYTSKKESCVGL'
    ,'Q03311':'MQTQHTKVTQTHFLLWILLLCMPFGKSHTEEDFIITTKTGRVRGLSMPVLGGTVTAFLGIPYAQPPLGSLRFKKPQPLNKWPDIHNATQYANSCYQNIDQAFPGFQGSEMWNPNTNLSEDCLYLNVWIPVPKPKNATVMVWIYGGGFQTGTSSLPVYDGKFLARVERVIVVSMNYRVGALGFLAFPGNPDAPGNMGLFDQQLALQWVQRNIAAFGGNPKSITIFGESAGAASVSLHLLCPQSYPLFTRAILESGSSNAPWAVKHPEEARNRTLTLAKFTGCSKENEMEMIKCLRSKDPQEILRNERFVLPSDSILSINFGPTVDGDFLTDMPHTLLQLGKVKKAQILVGVNKDEGTAFLVYGAPGFSKDNDSLITRKEFQEGLNMYFPGVSRLGKEAVLFYYVDWLGEQSPEVYRDALDDVIGDYNIICPALEFTKKFAELENNAFFYFFEHRSSKLPWPEWMGVMHGYEIEFVFGLPLGRRVNYTRAEEIFSRSIMKTWANFAKYGHPNGTQGNSTMWPVFTSTEQKYLTLNTEKSKIYSKLRAPQCQFWRLFFPKVLEMTGDIDETEQEWKAGFHRWSNYMMDWQNQFNDYTSKKESCTAL'
    ,'P81908':'EEDIIITTKNGKVRGMNLPVLGGTVTAFLGIPYAQPPLGRLRFKKPQSLTKWSNIWNATKYANSCYQNTDQSFPGFLGSEMWNPNTELSEDCLYLNVWIPAPKPKNATVMIWIYGGGFQTGTSSLPVYDGKFLARVERVIVVSMNYRVGALGFLALSENPEAPGNMGLFDQQLALQWVQKNIAAFGGNPRSVTLFGESAGAASVSLHLLSPRSQPLFTRAILQSGSSNAPWAVTSLYEARNRTLTLAKRMGCSRDNETEMIKCLRDKDPQEILLNEVFVVPYDTLLSVNFGPTVDGDFLTDMPDTLLQLGQFKRTQILVGVNKDEGTAFLVYGAPGFSKDNNSIITRKEFQEGLKIFFPRVSEFGRESILFHYMDWLDDQRAENYREALDDVVGDYNIICPALEFTRKFSELGNDAFFYYFEHRSTKLPWPEWMGVMHGYEIEFVFGLPLERRVNYTRAEEILSRSIMKRWANFAKYGNPNGTQNNSTRWPVFKSTEQKYLTLNTESPKVYTKLRAQQCRFWTLFFPKVLELTGNIDEAEREWKAGFHRWNNYMMDWKNQFNDYTSKKESCSDF'
    }

AchE_trained_Prot = {
        "P22303":'MRPPQCLLHTPSLASPLLLLLLWLLGGGVGAEGREDAELLVTVRGGRLRGIRLKTPGGPVSAFLGIPFAEPPMGPRRFLPPEPKQPWSGVVDATTFQSVCYQYVDTLYPGFEGTEMWNPNRELSEDCLYLNVWTPYPRPTSPTPVLVWIYGGGFYSGASSLDVYDGRFLVQAERTVLVSMNYRVGAFGFLALPGSREAPGNVGLLDQRLALQWVQENVAAFGGDPTSVTLFGESAGAASVGMHLLSPPSRGLFHRAVLQSGAPNGPWATVGMGEARRRATQLAHLVGCPPGGTGGNDTELVACLRTRPAQVLVNHEWHVLPQESVFRFSFVPVVDGDFLSDTPEALINAGDFHGLQVLVGVVKDEGSYFLVYGAPGFSKDNESLISRAEFLAGVRVGVPQVSDLAAEAVVLHYTDWLHPEDPARLREALSDVVGDHNVVCPVAQLAGRLAAQGARVYAYVFEHRASTLSWPLWMGVPHGYEIEFIFGIPLDPSRNYTAEEKIFAQRLMRYWANFARTGDPNEPRDPKAPQWPPYTAGAQQYVSLDLRPLEVRRGLRAQACAFWNRFLPKLLSATDTLDEAERQWKAEFHRWSSYMVHWKNQFDHYSKQDRCSDL',
        "P23795":'MRPPWCPLHTPSLTPPLLLLLFLIGGGAEAEGPEDPELLVMVRGGRLRGLRLMAPRGPVSAFLGIPFAEPPVGPRRFLPPEPKRPWPGVLNATAFQSVCYQYVDTLYPGFEGTEMWNPNRELSEDCLYLNVWTPYPRPSSPTPVLVWIYGGGFYSGASSLDVYDGRFLTQAEGTVLVSMNYRVGAFGFLALPGSREAPGNVGLLDQRLALQWVQENVAAFGGDPTSVTLFGESAGAASVGMHLLSPPSRGLFHRAVLQSGAPNGPWATVGVGEARRRATLLARLVGCPPGGAGGNDTELVACLRARPAQDLVDHEWRVLPQESVFRFSFVPVVDGDFLSDTPEALINAGDFHGLQVLVGVVKDEGSYFLVYGAPGFSKDNESLISRAQFLAGVRVGVPQASDLAAEAVVLHYTDWLHPEDPARLREALSDVVGDHNVVCPVAQLAGRLAAQGARVYAYIFEHRASTLSWPLWMGVPHGYEIEFIFGLPLEPSLNYTIEERTFAQRLMRYWANFARTGDPNDPRDPKAPQWPPYTAGAQQYVSLNLRPLEVRRGLRAQACAFWNRFLPKLLSATDTLDEAERQWKAEFHRWSSYMVHWKNQFDHYSKQDRCSDL',
        "P21836":'MRPPWYPLHTPSLAFPLLFLLLSLLGGGARAEGREDPQLLVRVRGGQLRGIRLKAPGGPVSAFLGIPFAEPPVGSRRFMPPEPKRPWSGVLDATTFQNVCYQYVDTLYPGFEGTEMWNPNRELSEDCLYLNVWTPYPRPASPTPVLIWIYGGGFYSGAASLDVYDGRFLAQVEGAVLVSMNYRVGTFGFLALPGSREAPGNVGLLDQRLALQWVQENIAAFGGDPMSVTLFGESAGAASVGMHILSLPSRSLFHRAVLQSGTPNGPWATVSAGEARRRATLLARLVGCPPGGAGGNDTELIACLRTRPAQDLVDHEWHVLPQESIFRFSFVPVVDGDFLSDTPEALINTGDFQDLQVLVGVVKDEGSYFLVYGVPGFSKDNESLISRAQFLAGVRIGVPQASDLAAEAVVLHYTDWLHPEDPTHLRDAMSAVVGDHNVVCPVAQLAGRLAAQGARVYAYIFEHRASTLTWPLWMGVPHGYEIEFIFGLPLDPSLNYTTEERIFAQRLMKYWTNFARTGDPNDPRDSKSPQWPPYTTAAQQYVSLNLKPLEVRRGLRAQTCAFWNRFLPKLLSATDTLDEAERQWKAEFHRWSSYMVHWKNQFDHYSKQERCSDL'        
    }

screen_data_process('screen_database/L6030.sdf',AchE_trained_Prot)