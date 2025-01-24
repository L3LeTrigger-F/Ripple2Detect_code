import os
import pandas as pd
import random
import csv
def build_user_item_ii():
    item_list=[0,1,2,3,4,5,6,7,8,9,10,11]
    user_item_list=[]
    r1=200
    i=0 #这都是单一行为的正经恶意用户，导入是前几个行为，预测最后一个行为。
    tens=[]
    while r1>0:
        user_item_list.append([0,i,0,1])#1就代表1！ 状态0
        user_item_list.append([0,i,8,1])
        xx=1
        while xx<=11:
            if xx!=8:
                user_item_list.append([0,i,xx,2])
            xx+=1
        i+=1
        r1-=1
    r1=200
    while r1>0:
        user_item_list.append([1,i,1,1])
        user_item_list.append([1,i,2,2])
        user_item_list.append([1,i,9,3])
        xx=0
        while xx<=11:
            if xx!=1 and xx!=2 and xx!=9:
                user_item_list.append([1,i,xx,4])
            xx+=1
        r1-=1
        i+=1
    r1=200
    while r1>0:
        user_item_list.append([2,i,1,1])
        user_item_list.append([2,i,2,1])
        user_item_list.append([2,i,9,1])
        xx=0
        while xx<=11:
            if xx!=1 and xx!=2 and xx!=9:
                user_item_list.append([2,i,xx,2])
            xx+=1
        r1-=1
        i+=1
    r1=200
    while r1>0:
        user_item_list.append([3,i,3,1])
        user_item_list.append([3,i,4,2])
        user_item_list.append([3,i,5,3])
        user_item_list.append([3,i,6,4])
        user_item_list.append([3,i,10,5])
        xx=0
        while xx<=11:
            if xx!=3 and xx!=4 and xx!=5 and xx!=6 and xx!=10:
                user_item_list.append([3,i,xx,6])
            xx+=1
        r1-=1
        i+=1
    r1=200
    while r1>0:
        user_item_list.append([4,i,3,1])
        user_item_list.append([4,i,4,1])
        user_item_list.append([4,i,5,2])
        user_item_list.append([4,i,6,3])
        user_item_list.append([4,i,10,4])
        xx=0
        while xx<=11:
            if xx!=3 and xx!=4 and xx!=5 and xx!=6 and xx!=10:
                user_item_list.append([4,i,xx,5])
            xx+=1
        r1-=1
        i+=1
    r1=200
    while r1>0:
        user_item_list.append([5,i,3,1])
        user_item_list.append([5,i,4,1])
        user_item_list.append([5,i,5,1])
        user_item_list.append([5,i,6,2])
        user_item_list.append([5,i,10,3])
        xx=0
        while xx<=11:
            if xx!=3 and xx!=4 and xx!=5 and xx!=6 and xx!=10:
                user_item_list.append([5,i,xx,4])
            xx+=1
        r1-=1
        i+=1
    r1=200
    while r1>0:
        user_item_list.append([6,i,3,1])
        user_item_list.append([6,i,4,1])
        user_item_list.append([6,i,5,1])
        user_item_list.append([6,i,6,1])
        user_item_list.append([6,i,10,1])
        xx=0
        while xx<=11:
            if xx!=3 and xx!=4 and xx!=5 and xx!=6 and xx!=10:
                user_item_list.append([6,i,xx,2])
            xx+=1
        r1-=1
        i+=1
    r1=200
    while r1>0:
        user_item_list.append([7,i,7,1])
        user_item_list.append([7,i,11,1])
        xx=0
        while xx<=11:
            if xx!=7 and xx!=11:
                user_item_list.append([7,i,xx,2])
            xx+=1
        r1-=1
        i+=1

            
    #单一乱点用户
    '''
    r4=1000
    item_list=[0,1,2,3,4,5,6,7,8]
    while r4>0:
        num=random.randint(0,8)
        choss=random.sample(item_list,num)
        if 0 in choss:
            if 6 not in choss:
                choss.append(6)
        if 1 in choss and 2 in choss:
            if 7 not in choss:
                choss.append(7)
        if 1 in choss and 2 not in choss:
            if 7 in choss:
                choss.remove(7) 
        if 4 in choss and 5 in choss:
            if 8 not in choss:
                choss.append(8)
        if 4 in choss and 5 not in choss:
            if 8 in choss:
                choss.remove(8)
        if  5 in choss and 4 not in choss:
            if 8 in choss:
                choss.remove(8)
        for item in choss:
            user_item_list.append([i,item,1])
        list_di=list(set(item_list) - set(choss)) 
        print(list_di)
        if list_di!=[]:
            if len(list_di)>=2:
                xx=random.sample(list_di,2)
            else:
                xx=random.sample(list_di,1)
            for a in xx:
                user_item_list.append([i,a,0])
        i+=1
        r4-=1
    '''
    with open("/home/c402/LhL/RippleNet/RippleNet/data/evidence/ratings_final_sort.txt","w") as f:
        for data in user_item_list:
            for d in data:
                f.write(str(d)+" ")
            f.write("\n")
# build_user_item_ii()    
def build_user_item():
    item_list=[0,1,2,3,4,5,6,7,8,9,10]
    user_item_list=[]
    r1=1000
    i=0 #这都是单一行为的正经恶意用户，导入是前几个行为，预测最后一个行为。
    tens=[]
    while r1>0:
        user_item_list.append([i,0,1])##这些是恶意用户
        user_item_list.append([i,7,1])##标签值
        xx=1
        tens=[]
        while xx<11:
            if xx!=7:
                tens.append([i,xx,0])
            xx+=1
        # user_item_list.extend(random.sample(tens,2))
        user_item_list.extend(tens)
        r1-=1
        i+=1
    r2=1000
    tens=[]
    while r2>0:
        if r2%2==0:
            user_item_list.append([i,1,1])#第i个用户，第1个证据
            user_item_list.append([i,2,1])
            user_item_list.append([i,8,1])##标签值
        else:
            user_item_list.append([i,1,1])#第i个用户，第1个证据
            user_item_list.append([i,2,0])
            user_item_list.append([i,8,0])##标签值
        xx=0
        tens=[]
        while xx<11:
            if xx!=8 and xx!=1 and xx!=2:              
                tens.append([i,xx,0])
            xx+=1
        # user_item_list.extend(random.sample(tens,3))
        user_item_list.extend(tens)
        r2-=1
        i+=1
    r3=1000
    while r3>0:#同时发生4，5才能发生8
        if r3%3==0:
            user_item_list.append([i,3,1])#第i个用户，第1个证据
            user_item_list.append([i,4,0])
            user_item_list.append([i,5,0])
            user_item_list.append([i,9,0])##标签值
        if r3%3==1:
            user_item_list.append([i,3,1])#第i个用户，第1个证据
            user_item_list.append([i,4,1])
            user_item_list.append([i,5,0])
            user_item_list.append([i,9,0])##标签值
        if r3%3==2:
            user_item_list.append([i,3,1])#第i个用户，第1个证据
            user_item_list.append([i,4,1])
            user_item_list.append([i,5,1])
            user_item_list.append([i,9,1])##标签值
        xx=0
        tens=[]
        while xx<11:
            if xx!=4 and xx!=5 and xx!=3 and xx!=9:
                tens.append([i,xx,0])
            xx+=1
        # user_item_list.extend(random.sample(tens,2)) 
        user_item_list.extend(tens)
        r3-=1
        i+=1
    r4=1000
    while r4>0:#同时发生4，5才能发生8
        user_item_list.append([i,6,1])#第i个用户，第1个证据
        user_item_list.append([i,10,1])
        xx=0
        tens=[]
        while xx<11:
            if xx!=6 and xx!=10:
                tens.append([i,xx,0])
            xx+=1
        # user_item_list.extend(random.sample(tens,2)) 
        user_item_list.extend(tens)
        r4-=1
        i+=1
    # r5=1000
    # while r5>0:
        # aa=random.sample([0,1] ,1)[0]
        # if aa==0:
            
    with open("/home/c402/LhL/RippleNet/RippleNet/data/evidence/ratings_final.txt","w") as f:
        for data in user_item_list:
            for d in data:
                f.write(str(d)+" ")
            f.write("\n")
def build_user_item_aa():
    item_list=[0,1,2,3,4,5,6,7,8,9,10]
    user_item_list=[]
    r1=4000
    i=0
    while(r1>0):
        aa=random.sample([0,1],1)[0]
        if aa==1:
            user_item_list.append([i,0,1])##这些是恶意用户
            user_item_list.append([i,7,1])##标签值
            xx=1
            tens=[]
            while xx<11:
                if xx!=7:
                    tens.append([i,xx,0])
                xx+=1
        # user_item_list.extend(random.sample(tens,2))
            user_item_list.extend(tens)
            r1-=1
            i+=1
        aa=random.sample([0,1],1)[0]
        if aa==1:
            user_item_list.append([i,1,1])#第i个用户，第1个证据
            user_item_list.append([i,2,1])
            user_item_list.append([i,8,1])##标签值
            xx=0
            tens=[]
            while xx<11:
                if xx!=8 and xx!=1 and xx!=2:              
                    tens.append([i,xx,0])
                xx+=1
        # user_item_list.extend(random.sample(tens,3))
            user_item_list.extend(tens)
            r1-=1
            i+=1
        aa=random.sample([0,1],1)[0]
        if aa==1:
            user_item_list.append([i,3,1])#第i个用户，第1个证据
            user_item_list.append([i,4,1])
            user_item_list.append([i,5,1])
            user_item_list.append([i,9,1])##标签值
            xx=0
            tens=[]
            while xx<11:
                if xx!=4 and xx!=5 and xx!=3 and xx!=9:
                    tens.append([i,xx,0])
                xx+=1
        # user_item_list.extend(random.sample(tens,2)) 
            user_item_list.extend(tens)
            r1-=1
            i+=1
        aa=random.sample([0,1],1)[0]
        if aa==1:
            user_item_list.append([i,6,1])#第i个用户，第1个证据
            user_item_list.append([i,10,1])
            xx=0
            tens=[]
            while xx<11:
                if xx!=6 and xx!=10:
                    tens.append([i,xx,0])
                xx+=1
        # user_item_list.extend(random.sample(tens,2)) 
            user_item_list.extend(tens)
            r1-=1
            i+=1
            
    with open("/home/c402/LhL/RippleNet/RippleNet/data/evidence/ratings_final.txt","w") as f:
        for data in user_item_list:
            for d in data:
                f.write(str(d)+" ")
            f.write("\n")
# build_user_item_ii()
# build_user_item()
def get_mal():
    mal_user=[]
    benign_user=[]
    with open("/home/c402/LhL/RippleNet/RippleNet/data/ratings_final.txt","r") as f:
        for line in f.readlines():
            user = line.strip().split(' ')[0]
            item = line.strip().split(' ')[1]
            label = line.strip().split(' ')[2]
            if label=='1' and (item=='6' or item=='7' or item=='8'):
                mal_user.append(user)
            else:
                if user not in benign_user:
                    benign_user.append(user)
    with open("/home/c402/LhL/RippleNet/RippleNet/data/user_label.txt","w") as f:
        for data in mal_user:
            f.write(str(data)+" "+str(1))
            f.write("\n")
        for data in benign_user:
            f.write(str(data)+" "+str(0))
            f.write("\n")
# get_mal()
def build_user_prediction():
    item_list=[0,1,2,3,4,5,6,7,8,9,10,11]
    user_item_list=[]
    r1=1000
    i=0 #这都是单一行为的正经恶意用户，导入是前几个行为，预测最后一个行为。
    tens=[]
    while r1>0:
        user_item_list.append([i,0,1])##这些是恶意用户
        user_item_list.append([i,8,1])##标签值
        xx=1
        tens=[]
        while xx<11:
            if xx!=8:
                tens.append([i,xx,0])
            xx+=1
        # user_item_list.extend(random.sample(tens,2))
        user_item_list.extend(tens)
        r1-=1
        i+=1
    r2=1000
    tens=[]
    while r2>0:
        user_item_list.append([i,1,1])#第i个用户，第1个证据
        user_item_list.append([i,2,1])
        user_item_list.append([i,9,1])##标签值
        xx=0
        tens=[]
        while xx<11:
            if xx!=9 and xx!=1 and xx!=2:               
                tens.append([i,xx,0])
            xx+=1
        # user_item_list.extend(random.sample(tens,3))
        user_item_list.extend(tens)
        r2-=1
        i+=1
    r3=1000
    while r3>0:#同时发生4，5才能发生8
        user_item_list.append([i,3,1])#第i个用户，第1个证据
        user_item_list.append([i,4,1])
        user_item_list.append([i,5,1])
        user_item_list.append([i,6,1])##标签值
        user_item_list.append([i,10,1])##标签值
        xx=0
        tens=[]
        while xx<11:
            if xx!=4 and xx!=5 and xx!=3 and xx!=10 and xx!=6:
                tens.append([i,xx,0])
            xx+=1
        # user_item_list.extend(random.sample(tens,4)) 
        user_item_list.extend(tens)
        r3-=1
        i+=1
    r4=1000
    while r4>0:#同时发生4，5才能发生8
        user_item_list.append([i,7,1])#第i个用户，第1个证据
        user_item_list.append([i,11,1])
        xx=0
        tens=[]
        while xx<11:
            if xx!=7 and xx!=11:
                tens.append([i,xx,0])
            xx+=1
        # user_item_list.extend(random.sample(tens,2)) 
        user_item_list.extend(tens)
        r4-=1
        i+=1
    #单一乱点用户
    '''
    r4=1000
    item_list=[0,1,2,3,4,5,6,7,8]
    while r4>0:
        num=random.randint(0,8)
        choss=random.sample(item_list,num)
        if 0 in choss:
            if 6 not in choss:
                choss.append(6)
        if 1 in choss and 2 in choss:
            if 7 not in choss:
                choss.append(7)
        if 1 in choss and 2 not in choss:
            if 7 in choss:
                choss.remove(7) 
        if 4 in choss and 5 in choss:
            if 8 not in choss:
                choss.append(8)
        if 4 in choss and 5 not in choss:
            if 8 in choss:
                choss.remove(8)
        if  5 in choss and 4 not in choss:
            if 8 in choss:
                choss.remove(8)
        for item in choss:
            user_item_list.append([i,item,1])
        list_di=list(set(item_list) - set(choss)) 
        print(list_di)
        if list_di!=[]:
            if len(list_di)>=2:
                xx=random.sample(list_di,2)
            else:
                xx=random.sample(list_di,1)
            for a in xx:
                user_item_list.append([i,a,0])
        i+=1
        r4-=1
    '''
    with open("/home/c402/LhL/RippleNet/RippleNet/data/evidence/ratings_final.txt","w") as f:
        for data in user_item_list:
            for d in data:
                f.write(str(d)+" ")
            f.write("\n")

build_user_prediction()
