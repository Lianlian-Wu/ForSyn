import pandas as pd
import numpy as np
import itertools
np.set_printoptions(suppress=True)


phychem_file="data20210125/drugfeature2_phychem_extract.csv"  
finger_file="data20210125/drugfeature1_finger_extract.csv"    
cell_line_path="data20210125/drugfeature3_express_extract/"   
cell_line_files=["A375", "A549", "BT20", "HCT116", "HS578T", "HT29", "LNCAP", "LOVO", "MCF7", "MDAMB231", "PC3", "RKO", "SKMEL28", "SW620", "VCAP"]
drugdrug_file="data20210125/drugdrug_extract.csv"  
cell_line_feature="data20210125/cell-line-feature_express_extract.csv"  

def load_data(cell_line_name="all",score="S",dataid=8):   
    extract=pd.read_csv(drugdrug_file,usecols=[3,4,5,11]) 
    phychem=pd.read_csv(phychem_file)
    finger=pd.read_csv(finger_file)
    cell_line=pd.read_csv(cell_line_feature)
    column_name=list(finger.columns)      
    column_name[0]="drug_id"
    finger.columns=column_name 
    label=pd.Categorical(extract["label"])   
    extract["label"]=label.codes+1  
    #print(list(cell_line["A375"]))
    
    if cell_line_name=="all":
        all_express={cell_line:pd.read_csv("{}{}.csv".format(cell_line_path,cell_line)) for cell_line in cell_line_files}
    elif type(cell_line_name) is list:
        all_express={cell_line:pd.read_csv("{}{}.csv".format(cell_line_path,cell_line)) for cell_line in cell_line_name}
    elif type(cell_line_name) is str:
        all_express={cell_line_name:pd.read_csv("{}{}.csv".format(cell_line_path,cell_line_name))}
    else:
        raise ValueError("Invalid parameter: {}".format(cell_line_name))    
    
    drug_comb=None
    if cell_line_name=="all":
        drug_comb=extract
    else:
        if type(cell_line_name) is list:
            drug_comb=extract.loc[extract["cell_line_name"].isin(cell_line_name)]
        else:
            drug_comb=extract.loc[extract["cell_line_name"]==cell_line_name]

    n_sample=drug_comb.shape[0]
    n_feature=((phychem.shape[1]-1)+(finger.shape[1]-1)+978)*2+1+978
    drug_comb.index=range(n_sample)
    #data=np.zeros((n_sample,n_feature))
    data=[]
    dataid=dataid
    for i in range(n_sample):
        drugA_id=drug_comb.at[i,"drug_row_cid"]
        drugB_id=drug_comb.at[i,"drug_col_cid"]
        drugA_finger=get_finger(finger,drugA_id)
        drugB_finger=get_finger(finger,drugB_id)
        drugA_phychem=get_phychem(phychem,drugA_id)
        drugB_phychem=get_phychem(phychem,drugB_id)
        cell_line_name=drug_comb.at[i,"cell_line_name"]
        drugA_express=get_express(all_express[cell_line_name],drugA_id)
        drugB_express=get_express(all_express[cell_line_name],drugB_id)
        feature=get_cell_feature(cell_line,cell_line_name)
        label=drug_comb.at[i,"label"]
        
        if dataid==1:
            sample=np.hstack((drugA_finger,drugB_finger,feature,label))                             #data1_2740 
        elif dataid==2:
            sample=np.hstack((drugA_phychem,drugB_phychem,feature,label))                           #data2_1088 
        elif dataid==3:
            sample=np.hstack((drugA_express,drugB_express,label))                                   #data3_1956 
        elif dataid==4:
            sample=np.hstack((drugA_finger,drugA_phychem,drugB_finger,drugB_phychem,feature,label)) #data4_2850 
        elif dataid==5:
            sample=np.hstack((drugA_finger,drugA_express,drugB_finger,drugB_express,label))         #data5_3718         
        elif dataid==6:
            sample=np.hstack((drugA_phychem,drugA_express,drugB_phychem,drugB_express,label))       #data6_2066 
        elif dataid==7:
            sample=np.hstack((drugA_finger,drugA_phychem,drugA_express,drugB_finger,drugB_phychem,drugB_express,label))          #data7_3828 
        elif dataid==8:
            sample=np.hstack((drugA_finger,drugA_phychem,drugA_express,drugB_finger,drugB_phychem,drugB_express,feature,label))  #data8_4806      
        data.append(sample)
    print("***************load data-{}***************".format(dataid))
    data=np.array(data)
    return data[:,0:-1],data[:,-1],dataid
    
def get_finger(finger,drug_id):
    drug_finger=finger.loc[finger['drug_id']==drug_id]
    drug_finger=np.array(drug_finger)
    drug_finger=drug_finger.reshape(drug_finger.shape[1])[1:]
    # print(drug_finger.shape)
    return drug_finger

def get_phychem(phychem,drug_id):
    drug_phychem=phychem.loc[phychem["cid"]==drug_id]
    # print(drug_phychem)
    drug_phychem=np.array(drug_phychem)
    drug_phychem=drug_phychem.reshape(drug_phychem.shape[1])[1:]
    # print(drug_phychem.shape)
    return drug_phychem

def get_express(express,drug_id):
    if str(drug_id) not in express.columns.values:
        return None
    drug_express=express[str(drug_id)]
    drug_express=np.array(drug_express)
    # print(drug_express.shape)
    return drug_express

def get_cell_feature(feature,cell_line_name):
    # print(feature.head())
    # print(cell_line_name)
    cell_feature=feature[str(cell_line_name)]
    cell_feature=np.array(cell_feature)
    return cell_feature

def get_drugs():
    drugdrug=pd.read_csv(drugdrug_file,usecols=[4,5])
    drugAs=drugdrug["drug_row_cid"]
    drugBs=drugdrug["drug_col_cid"]
    drugs=np.hstack((np.array(drugAs),np.array(drugBs)))
    drugs=np.unique(drugs)
    return drugs

def get_drugdrug():
    drugs=get_drugs()
    drugdrugs=list(itertools.combinations(drugs,2))
    # drugdrugs=np.array(drugdrugs)
    return drugdrugs

def get_all_samples():
    drugdrugs=get_drugdrug()
    cell_line=["A-673","A375","A549","HCT116","HS 578T","HT29","LNCAP","LOVO","MCF7","PC-3","RKO","SK-MEL-28","SW-620","VCAP"]
    all_drugdrugs=[]
    for i in range(len(drugdrugs)):
        for j in range(len(cell_line)):
            drugdrug=list(drugdrugs[i])
            drugdrug.append(cell_line[j])
            all_drugdrugs.append(drugdrug)
    return all_drugdrugs
            
def is_equal(sample1,sample2):
    if sample1[2]!=sample2[2]:
        return False
    elif (sample1[0]==sample2[0] and sample1[1]==sample2[1]) or (sample1[0]==sample2[1] and sample1[1]==sample2[0]):
        return True
    else:
        return False

def filter_no_expression():
    all_drugdrugs=np.array(get_all_samples())

    expression={cell_line:pd.read_csv("{}{}.csv".format(cell_line_path,cell_line)) for cell_line in cell_line_files}
    
    index=[]
    for i in range(len(all_drugdrugs)):
        drugA_id=all_drugdrugs[i][0]
        drugB_id=all_drugdrugs[i][1]
        cell_line=all_drugdrugs[i][2]
        drugA_express=get_express(expression[cell_line],drugA_id)
        drugB_express=get_express(expression[cell_line],drugB_id)
        if drugA_express is None or drugB_express is None:
            continue
        index.append(i)
    all_drugdrugs=all_drugdrugs[index]
    return all_drugdrugs

def get_unkonwn_drugdrug():
    all_drugdrugs=filter_no_expression()
    drugdrug3014=pd.read_csv(drugdrug_file,usecols=[3,4,5])
    drugdrug3014=drugdrug3014[["drug_row_cid","drug_col_cid","cell_line_name"]]
    index=[]
    for i in range(all_drugdrugs.shape[0]):
        drugdrug=all_drugdrugs[i]
        if_exist=False
        if i%200==0:
            print("=====")
        for j in range(drugdrug3014.shape[0]):
            tmp=drugdrug3014.iloc[j]
            tmp=[str(tmp[k]) for k in range(3)]
            # print(tmp)
            # print(drugdrug)
            # input()
            if is_equal(drugdrug,tmp):
                if_exist=True
                break
        if if_exist==False:
            index.append(i)
    return all_drugdrugs[index]

def load_unknown_data():

    phychem=pd.read_csv(phychem_file)
    finger=pd.read_csv(finger_file)
    cell_feature=pd.read_csv(cell_line_feature)
    column_name=list(finger.columns)
    column_name[0]="drug_id"
    finger.columns=column_name
    gene_expressions={cell_line:pd.read_csv("{}{}.csv".format(cell_line_path,cell_line)) for cell_line in cell_line_files}
    drugdrugs=pd.read_csv("data0825/data/unknown_drugdrug.csv")

    data=np.zeros((drugdrugs.shape[0],4806))
    for i in range(data.shape[0]):
        drugA_id=drugdrugs.at[i,"drug_row_cid"]
        drugB_id=drugdrugs.at[i,"drug_col_cid"]
        cell_line=drugdrugs.at[i,"cell_line_name"]
        feature=get_cell_feature(cell_feature,cell_line)
        drugA_finger=get_finger(finger,drugA_id)
        drugB_finger=get_finger(finger,drugB_id)
        drugA_phychem=get_phychem(phychem,drugA_id)
        drugB_phychem=get_phychem(phychem,drugB_id)
        drugA_express=get_express(gene_expressions[cell_line],drugA_id)
        drugB_express=get_express(gene_expressions[cell_line],drugB_id)
        sample=np.hstack((drugA_finger,drugA_phychem,drugA_express,drugB_finger,drugB_phychem,drugB_express,feature))
        data[i]=sample
    return data

#x,y=load_data()
#print(x.shape)
#print(y.shape)