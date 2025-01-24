import pandas as pd
import time
from datetime import datetime
import os
import json
import h5py
import numpy as np
#案例一：
#案例二：用户开始浏览招聘网站并向竞争对手招揽工作。在离开公司之前，他们使用拇指驱动器（其速度明显高于以前的活动）来窃取数据。

#案例三：系统管理员对工作不满意，在领导的计算机上安装某个键盘记录器，记录密码，作为领导登录邮箱，发送大量电子邮件损害公司利益，迅速离开公司
    #攻击表现：以领导身份在领导的电脑发送了大量电子邮件损害公司利益。
    #成立需要：1、aa下载了键盘钩子。2、在领导的计算机上安装某个键盘记录器记录密码 3、事前查看记录的密码是什么
    #混淆：          【】                         【非领导】                           【事前没查看】 （针对对比试验设计）
#案例四：用户登录到其他人的设备时，将搜索相关文件，并通过公司邮箱将其发送到家庭邮箱。这种行为将在未来持续一段时间。
    #攻击表现：
    #成立需要：1、用户登录到其他人的设备  2、窃取文件and发家庭邮箱
    #混淆：                                    【没发家庭邮箱】
#读取每个案例的起始时间，取操作者起始时间内的操作
def convert(json_name):
    path="/disk1/lhl/cross_certification/dataset/act_sequence"
    file_name=json_name[:-5]
    json_name=os.path.join(path,json_name)
    with open(json_name,"r") as f:  
        json_data=json.load(f)
        df = pd.DataFrame(json_data)
        path=os.path.join(path,file_name+".h5")
        df.to_hdf(path,key="email")
def get_case(case_path,scene):
    la_info=[]
    for file in os.listdir(case_path):
        fp=case_path+"/"+file
        ##行数不一致
        largest_column_count =0
        with open(fp, 'r') as temp_f:
            lines = temp_f.readlines()
            for l in lines:
                column_count = len(l.split(',')) + 1
        #找到列数最多的行
                largest_column_count = column_count if largest_column_count < column_count else largest_column_count
        temp_f.close()
        
        # colunm_names为最大列数展开
        column_names = [i for i in range(0, largest_column_count)]
        case_info=pd.read_csv(fp,header=None, delimiter=',', names=column_names)
        sta=case_info.values[0,2]
        ends=case_info.values[-1,2]
        #日志分为三部分：下属用本机操作，上司用他的电脑操作，下属用上司的电脑操作
        n1=case_info.values[0,3]
        n2=case_info.values[1,3]
        p1=case_info.values[0,4]
        p2=case_info.values[1,4]
        ope={"yee":n1,"er":n2,"p1":p1,"p2":p2,"sta":sta,"ends":ends}
        la_info.append(ope)
    return la_info
def append_data_to_h5(file_path, dataset_name, data_val,data_time):
    objects_data=np.array(data_val, dtype='S')
    object_time=np.array(data_time, dtype='S')
    with h5py.File(file_path, 'a') as hf:
        if dataset_name+"_data" in hf:
            del hf[dataset_name+"_data"]
            hf[dataset_name+"_data"]=objects_data
        else:
            hf.create_dataset(dataset_name+"_data", data=objects_data)
        if dataset_name+"_time" in hf:
            del hf[dataset_name+"_time"]
            hf[dataset_name+"_time"]=object_time
        else:
            hf.create_dataset(dataset_name+"_time", data=object_time)
def get_email(email,la_info,new_path):
    all_dict_val={}
    all_dict_time={}
    all_dict={}
    name_info=[]
    sta_dict={}
    end_dict={}
    op=0
    for la in la_info:
        all_dict_val[la]=[]
        all_dict_time[la]=[]
    # all_dict=
    # for index,row in email.iterrows():
        # if row['date']=="09/15/2010 07:42:02":
            # op=index
            # break
    # for la in la_info:
        #name_info.append(la["yee"])
        #sentence=""
        #sta_dict[la["yee"]]=int(time.mktime(time.strptime(la["sta"],"%m/%d/%Y %H:%M:%S")))
        #end_dict[la["yee"]]=int(time.mktime(time.strptime(la["ends"],"%m/%d/%Y %H:%M:%S")))
    for index,row in email.iterrows():
        if row['user'] in la_info:
            iter_sta=int(time.mktime(time.strptime(row["date"],"%m/%d/%Y %H:%M:%S")))
                #sta_stamp=sta_dict[row['user']]
                #ends_stamp=end_dict[row['user']]
                #if iter_sta<sta_stamp:
                #    continue
                #if iter_sta>ends_stamp:
                #    continue
            sentence=""
            sentence+=(row['user']+" ")
            sentence+=(row['pc']+" ")
            sentence+=(row['to']+" ")
            sentence+=(row['content']+";")
            all_dict_val[row['user']].append(sentence)
            all_dict_time[row['user']].append(iter_sta)
    for la in la_info:
        path=os.path.join("/home/c402/LhL/cross_certification/dataset/r5.2/mal_sec",la+".h5")
        append_data_to_h5( path, "email",all_dict_val[la],all_dict_time[la])
                #all_dict[row['user']].append({"timestamp":iter_sta,"sens":sentence})
                #all_dict.append({"timestamp":iter_sta,"sens":sentence})
                # json_name=new_path+row['user']+".json"
                # if os.path.exists(json_name):
                    # with open(json_name,"r") as f:  
                        # json_data=json.load(f)
                        # json_data["email"].append({"timestamp":iter_sta,"sens":sentence})
                    # with open(json_name,"w") as f_new:        
                        # json.dump(json_data,f_new,indent=4)
                # else:
                    # with open(json_name,"w") as f_new:        
                
                        # json.dump({"email":[{"timestamp":iter_sta,"sens":sentence}]},f_new,indent=4)
    #for la in la_info:
    #    json_name=new_path+la["yee"]+".json"
    #    if os.path.exists(json_name):
    #        with open(json_name,"r") as f:  
    #            json_data=json.load(f)
    #            json_data["email"]=all_dict[la["yee"]]
    #        with open(json_name,"w") as f_new:        
    #            json.dump(json_data,f_new,indent=4)
    #    else:
    #        with open(json_name,"w") as f: 
    #            json.dump({"email":all_dict[la["yee"]]},f,indent=4)
def get_logon(logon,la_info,new_path):
    all_dict_val={}
    all_dict_time={}
    name_info=[]
    sta_dict={}
    end_dict={}
    for la in la_info:
        all_dict_val[la]=[]
        all_dict_time[la]=[]
        #all_dict[la["yee"]]=[]
        # all_dict=[]
        #name_info.append(la["yee"])
        sentence=""
        #sta_dict[la["yee"]]=int(time.mktime(time.strptime(la["sta"],"%m/%d/%Y %H:%M:%S")))
        #end_dict[la["yee"]]=int(time.mktime(time.strptime(la["ends"],"%m/%d/%Y %H:%M:%S")))
    for index,row in logon.iterrows():
        if row['user'] in la_info:
            iter_sta=int(time.mktime(time.strptime(row["date"],"%m/%d/%Y %H:%M:%S")))
                #sta_stamp=sta_dict[row['user']]
                #ends_stamp=end_dict[row['user']]
                #if iter_sta<sta_stamp:
                #    continue
                #if iter_sta>ends_stamp:
                #    continue
            sentence=""
            sentence+=(row['user']+" ")
            sentence+=(row['pc']+" ")
            sentence+=(row['activity']+";")
            all_dict_val[row['user']].append(sentence)
            all_dict_time[row['user']].append(iter_sta)
                #all_dict[row['user']].append({"timestamp":iter_sta,"sens":sentence})
    #for la in la_info:
    '''
                json_name=new_path+row['user']+".json"
                with open(json_name,"r") as f:  
                    json_data=json.load(f)
                    if "logon" in json_data:
                        json_data["logon"].append({"timestamp":iter_sta,"sens":sentence})
                    else:
                        json_data["logon"]={"timestamp":iter_sta,"sens":sentence}
                with open(json_name,"w") as f_new:        
                    json.dump(json_data,f_new,indent=4)
    '''
    for la in la_info:
        path=os.path.join("/home/c402/LhL/cross_certification/dataset/r5.2/mal_sec",la+".h5")
        append_data_to_h5( path, "logon",all_dict_val[la],all_dict_time[la])
def get_time(name,logon):
    num=0
    st=0
    sta=""
    ends=""
    for index,row in logon.iterrows():
        if st!=1 and row['user']==name and row['activity']=="Logon":
            sta=row['date']
            st=1
        if row['user']==name and row['activity']=="Logoff":
            num+=1
            if num==15:
                ends=row['date']
                break
    return {"sta":sta,"ends":ends}

        
def get_files(files,la_info,new_path):
    all_dict_val={}
    all_dict_time={}
    name_info=[]
    sta_dict={}
    end_dict={}
    for la in la_info:
        all_dict_val[la]=[]
        all_dict_time[la]=[]
       # all_dict[la["yee"]]=[]
       # name_info.append(la["yee"])
       # sentence=""
       # sta_dict[la["yee"]]=int(time.mktime(time.strptime(la["sta"],"%m/%d/%Y %H:%M:%S")))
       # end_dict[la["yee"]]=int(time.mktime(time.strptime(la["ends"],"%m/%d/%Y %H:%M:%S")))
    for index,row in files.iterrows():
        if row['user'] in la_info:
            iter_sta=int(time.mktime(time.strptime(row["date"],"%m/%d/%Y %H:%M:%S")))
            #sta_stamp=sta_dict[row['user']]
            #ends_stamp=end_dict[row['user']]
            #if iter_sta<sta_stamp:
            #    continue
            #if iter_sta>ends_stamp:
            #   continue
            sentence="" 
            sentence+=(row['user']+" ")
            sentence+=(row['pc']+" ")
            sentence+=(row['filename']+" ")
            sentence+=(row['content']+";")
            all_dict_val[row['user']].append(sentence)
            all_dict_time[row['user']].append(iter_sta)
    for la in la_info:
        path=os.path.join("/home/c402/LhL/cross_certification/dataset/r5.2/mal_sec",la+".h5")
        append_data_to_h5( path, "files",all_dict_val[la],all_dict_time[la])
                #all_dict[row['user']].append({"timestamp":iter_sta,"sens":sentence})
    #for la in la_info:
    '''
            json_name=new_path+la["yee"]+".json"
            with open(json_name,"r") as f:  
                json_data=json.load(f)
                if "files" in json_data:
                    json_data["files"].append({"timestamp":iter_sta,"sens":sentence})
                else:
                    json_data["files"]={"timestamp":iter_sta,"sens":sentence}
                #json_data["files"]=all_dict[la["yee"]]
            with open(json_name,"w") as f_new:        
                json.dump(json_data,f_new,indent=4)
    '''
def get_device(device,la_info,new_path):
    all_dict={}
    all_dict_val={}
    all_dict_time={}
    name_info=[]
    sta_dict={}
    end_dict={}
    for la in la_info:
        all_dict_val[la]=[]
        all_dict_time[la]=[]
        #name_info.append(la["yee"])
        #sentence=""
        #sta_dict[la["yee"]]=int(time.mktime(time.strptime(la["sta"],"%m/%d/%Y %H:%M:%S")))
        #end_dict[la["yee"]]=int(time.mktime(time.strptime(la["ends"],"%m/%d/%Y %H:%M:%S")))
    for index,row in device.iterrows():
        if row['user'] in la_info:
            iter_sta=int(time.mktime(time.strptime(row["date"],"%m/%d/%Y %H:%M:%S")))
                #sta_stamp=sta_dict[row['user']]
                #ends_stamp=end_dict[row['user']]
            #if iter_sta<sta_stamp:
            #    continue
            #if iter_sta>ends_stamp:
            #   continue
            sentence=""
            sentence+=(row['user']+" ") 
            sentence+=(row['pc']+" ")
            sentence+=(row['activity']+";")
            all_dict_val[row['user']].append(sentence)
            all_dict_time[row['user']].append(iter_sta)
                #all_dict[row['user']].append({"timestamp":iter_sta,"sens":sentence})
    #for la in la_info:
    '''
            json_name=new_path+la["yee"]+".json"
            with open(json_name,"r") as f:  
                json_data=json.load(f)
                #json_data["device"]=all_dict[la["yee"]]
                if "device" in json_data:
                    json_data["device"].append({"timestamp":iter_sta,"sens":sentence})
                else:
                    json_data["device"]={"timestamp":iter_sta,"sens":sentence}
            with open(json_name,"w") as f_new:        
                json.dump(json_data,f_new,indent=4)
    '''
    for la in la_info:
        path=os.path.join("/home/c402/LhL/cross_certification/dataset/r5.2/mal_sec",la+".h5")
        append_data_to_h5( path, "device",all_dict_val[la],all_dict_time[la])
            
def get_http(http,la_info,new_path):
    all_dict={}
    all_dict_val={}
    all_dict_time={}
    name_info=[]
    sta_dict={}
    end_dict={}
    for la in la_info:
        all_dict_val[la]=[]
        all_dict_time[la]=[]
        #name_info.append(la["yee"])
        #sta_dict[la["yee"]]=int(time.mktime(time.strptime(la["sta"],"%m/%d/%Y %H:%M:%S")))
        #end_dict[la["yee"]]=int(time.mktime(time.strptime(la["ends"],"%m/%d/%Y %H:%M:%S")))
    sentence=""
    for ht in http:
        for index,row in ht.iterrows():
            if row['user'] in la_info:
                iter_sta=int(time.mktime(time.strptime(row["date"],"%m/%d/%Y %H:%M:%S")))
                  #  sta_stamp=sta_dict[row['user']]
                  #  ends_stamp=end_dict[row['user']]
                  #  if iter_sta<sta_stamp:
                 #       continue
               #     if iter_sta>ends_stamp:
                #        continue
                sentence=""
                sentence+=(row['user']+" ") 
                sentence+=(row['pc']+" ")
                sentence+=(row['url']+" ")
                sentence+=(row['content']+";")
                all_dict_val[row['user']].append(sentence)
                all_dict_time[row['user']].append(iter_sta)
                #all_dict[row['user']].append({"timestamp":iter_sta,"sens":sentence})
    #for la in la_info:
    '''
                json_name=new_path+la+".json"
                with open(json_name,"r") as f:  
                    json_data=json.load(f)
                    json_data["http"]=all_dict[la["yee"]]
                with open(json_name,"w") as f_new:        
                    json.dump(json_data,f_new,indent=4)
    '''
    for la in la_info:
        path=os.path.join("/home/c402/LhL/cross_certification/dataset/r5.2/mal_sec",la+".h5")
        append_data_to_h5( path, "http",all_dict_val[la],all_dict_time[la])
def read_csv(case_path,scene,modes):
    #time is:
    #if modes=="normal":
         #思路：1、建时序序列，每15个logon-off操作一个序列。
    #la_info=get_case(case_path)
    #    logon=pd.read_csv("/disk1/lhl/cross_certification/dataset/dialogs/logon.csv",usecols=['user','date','activity'])
    #    nn=list(logon.values)
    #    name=[]
    #    for dd in nn:
    #        name.append(dd[1])
    #    name=list(set(name))
     #   la_info=[]
     #   for na in name:
    #        print(na)
    #        times=get_time(na,logon)
    #        la_info.append({"yee":na,"sta":times["sta"],"ends":times["ends"]})
    #else:
       # la_info=get_case(case_path,scene)
    logon=pd.read_csv("/home/c402/LhL/cross_certification/dataset/r5.2/dialogs/logon.csv",usecols=['date','user','pc','activity'])
    user_list=list(set(list(logon['user'])))
    emails=pd.read_csv("/home/c402/LhL/cross_certification/dataset/r5.2/dialogs/email.csv",usecols=['date','user','pc','to','content'])
 #   files=pd.read_csv("/home/c402/LhL/cross_certification/dataset/r5.2/dialogs/file.csv",usecols=['date','user','pc','filename','content'])
 #   device=pd.read_csv("/home/c402/LhL/cross_certification/dataset/r5.2/dialogs/device.csv",usecols=['date','user','pc','activity'])  
 #   http=pd.read_csv("/home/c402/LhL/cross_certification/dataset/r5.2/dialogs/http.csv",usecols=['date','user','pc','url','content'],chunksize=10000,iterator = True)
    new_path="/home/c402/LhL/cross_certification/dataset/r5.2/mal_sec/"
    chunk_list=[]
    for ch in http:
        chunk_list.append(ch)
    #(21979935,5)
    if scene=='r3':
        print("--------email--------")
        get_email(emails,user_list,new_path)
  #      print("--------logon--------")
  #      get_logon(logon,user_list,new_path)
  #      print("--------files--------")
  #      get_files(files,user_list,new_path)
  #      print("--------device--------")
  #      get_device(device,user_list,new_path)
  #      print("--------http--------")
  #      get_http(chunk_list,user_list,new_path)
    elif scene=='r4':
        print("1")
#for file in os.listdir("/disk1/lhl/cross_certification/dataset/act_sequence"):
#    if file[-4:]=="json":
        # os.remove("/disk1/lhl/cross_certification/dataset/act_sequence/"+file)
# for file in os.listdir("/disk1/lhl/cross_certification/dataset/act_sequence"):
    # convert(file)

        



#create_normal_sequence()
#每15个logon为一组。
# f = h5py.File('/disk1/lhl/cross_certification/dataset/act_sec/ZSL0305.h5','r')
# print(f.keys())     
read_csv("/disk3/LhL/cross_certification/dataset/answers",'r3','normal')
    

#建立未知词表    
    