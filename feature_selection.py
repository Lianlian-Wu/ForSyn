import numpy as np
import itertools

def calc_f3(x,y):
    labels=np.unique(y)
    indexs={}
    c_mins={}
    c_maxs={}
    for label in labels:
        index=np.where(y==label)[0]
        indexs[label]=index
        c_min=np.min(x[index])
        c_max=np.max(x[index])
        c_mins[label]=c_min
        c_maxs[label]=c_max
    label_combin=list(itertools.combinations(labels,2))
    f3=0.0
    for combination in label_combin:
        # sample_num=len(indexs[combination[0]])+len(indexs[combination[1]])
        # print(sample_num)
        # print(combination)
        # print(sample_num)
        c1_max,c1_min=c_maxs[combination[0]],c_mins[combination[0]]
        c2_max,c2_min=c_maxs[combination[1]],c_mins[combination[1]]
        # print(c1_max,c1_min,c2_max,c2_min)
        if c1_max<c2_min or c2_max<c1_min:
            f3+=1
        else:
            interval=(max(c1_min,c2_min),min(c1_max,c2_max))
            sample=np.hstack((x[indexs[combination[0]]],x[indexs[combination[1]]]))
            # print(sample.shape[0])
            n_overlay=0
            for k in range(sample.shape[0]):
                if sample[k]>=interval[0] and sample[k]<=interval[1]:
                    n_overlay+=1
            f3+=1-n_overlay/sample.shape[0]
    f3/=len(label_combin)
    return f3

def select_feature(x,y,k):
    n_feature=x.shape[1]
    f3s=[0.0 for i in range(n_feature)]
    for i in range(n_feature):
        if len(np.unique(x[:,i]))==1:
            f3s[i]=0
        elif len(np.unique(x[:,i]))==2:
            f3s[i]=1
        else:
            f3s[i]=calc_f3(x[:,i],y)
    index=np.argsort(f3s)
    index=index[-k:]
    # return x[:,index],y
    return index


# x=np.array([[11,23,14,41,35], #0
#             [23,12,35,46,55], #2
#             [45,98,27,53,24], #0
#             [33,77,23,86,54], #0
#             [75,12,31,56,89], #1
#             [42,15,78,43,78], #1
#             [96,57,34,67,24], #0
#             [45,22,88,34,65], #2
#             [63,12,67,96,13]  #2
#             ])

# y=np.array([0,2,0,0,1,1,0,2,2])

# f3=calc_f3(x[:,2],y)
# print(f3)
# a=[9,3,43,54,36,77,68,235,8]
# b=np.argsort(a)
# print(b[-3:])