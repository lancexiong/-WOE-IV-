# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:08:04 2019

@author: lancexiong
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

class Binning(object):
    def __init__(self,confidenceVal=3.841, bins=5):
        self.confidenceVal=confidenceVal
        self.bins=bins
        self.cut_dict={}
        self.var_type={}
      #  self.label=None
    def add_miss2na(self,df,label):
        df_var=df.drop(label,axis=1)
        df_var=df_var.fillna('missing')
        return df_var
    def cut(self,df,variable,bins=20,cut_list=None):        
        '''
        return : 切分好的数据集df_cut和切分点列表cut_value
        df : 数据集
        variable : 待切分变量/特征
        '''
        if bins and (cut_list is not None):
            raise Exception("同时限定了bins和cut_list参数")
        if cut_list is None:
            df_cut=pd.qcut(df[variable],bins,duplicates='drop')
            df_cut=df_cut.dropna()
            df_temp=df_cut.drop_duplicates()
            cut_off=df_temp.sort_values()
            cut_off=cut_off.reset_index()
            cut_off.drop(['index'],axis=1,inplace=True)
            cut_value=[]
            for i in range(len(cut_off)):
                if i ==0:
                    cut_value.append(np.float('-inf'))
                else:
                    cut_value.append(cut_off.loc[i][0].left)
            cut_value.append(np.float('inf'))
            
            #self.cut_off=cut_value
            return df_cut,cut_value
        else:
            df_cut=pd.cut(df[variable],cut_list)
            return df_cut
        
    def cut_transform(self,df,cut_list):
        '''
        return : 分箱后的数据集
        df : 待分箱数据集
        cut_list : 切分规则，字典形式
        '''
        assert(type(cut_list)==dict),"切分规则不是字典形式"
        df_new=pd.DataFrame()
        for var in cut_list.keys():
			if is_string_dtype(df[var]):
				df_new[var]=df[var]
				continue
            cut_cri=list(cut_list[var]['breaks'])
            cut_cri.insert(0,np.float('-inf'))
            if cut_cri[-1]=='missing':
                cut_cri=cut_cri[:-1]
                df_new[var]=pd.cut(df[var],cut_cri)
            else:
                df_new[var]=pd.cut(df[var],cut_cri)
            df_new[var]=df_new[var].cat.add_categories("missing").fillna("missing")
        #df_new=df_new.merge(df[label],left_index=True,right_index=True,how='inner')
        return df_new
    def chi2(self,arr):
        '''
        return : 返回计算得到的卡方值
        arr : 数组形式:2*2维,行为切分区间，列为好坏客户
        '''
        error=0
        arr=arr.astype(np.float)
        #print("当前计算array: {} \n".format(arr))
        #print("当前的shape: {}".format(arr.shape))
        
        assert(arr.shape==(2,2)),"计算卡方值的array的shape不是2*2"
        ad=arr[0,0]*arr[1,1]
        bc=arr[0,1]*arr[1,0]
        n=np.sum(arr)
        row_sum=np.sum(arr,axis=0)
        col_sum=np.sum(arr,axis=1)
#        if (np.prod(row_sum)*np.prod(col_sum))==0:
#            print("分母为0，array为{}".format(arr))
        if np.prod(row_sum)==0 or np.prod(col_sum)==0:
            error=1
        chi=n*np.power((ad-bc),2)/(np.prod(row_sum)*np.prod(col_sum))
        return chi,error
    def ChiMerge(self,df, variable, flag, bins=5,confidenceVal=3.841,sample = None):  
        '''
        return : 返回合并完成后每个区间的卡方值和切分点列表
        df : 数据集
        variable : 待切分变量/特征
        flag : 目标变量label,离散类型
        confidenceVal : 阈值
        bin : 分箱数量
        sample : 是否抽样
        '''
    #进行是否抽样操作
        if sample != None:
            df = df[[variable,flag]].sample(n=sample)
        else:
            df   
    #先用等深粗略切分成20分箱，在考虑合并
        df_cut,cut_value=self.cut(df,variable)
        df_chi=pd.merge(df_cut,df[flag],left_index=True,right_index=True,how='inner')
        df_chi.columns=[variable,flag]
    #进行数据格式化录入
        total_num = df_chi.groupby([variable])[flag].count()  # 统计需分箱变量每个值数目
        total_num = pd.DataFrame({'total_num': total_num})  # 创建一个数据框保存之前的结果
        positive_class = df_chi.groupby([variable])[flag].sum()  # 统计需分箱变量每个值正样本数
        positive_class = pd.DataFrame({'positive_class': positive_class})  # 创建一个数据框保存之前的结果
        regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True,
                           how='inner')  # 组合total_num与positive_class
        regroup.reset_index(inplace=True)
        regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']  # 统计需分箱变量每个值负样本数
        regroup = regroup.drop('total_num', axis=1)
        np_regroup = np.array(regroup)  # 把数据框转化为numpy（提高运行效率）
        print('已完成变量{}数据读入,正在计算数据初处理'.format(variable))
        #处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
        i = 0
        while (i <= np_regroup.shape[0] - 2):
            if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or ( np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
                np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
                np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
                np_regroup[i, 0] = np_regroup[i + 1, 0]
                np_regroup = np.delete(np_regroup, i + 1, 0)
                i = i - 1
            i = i + 1
        #对相邻两个区间进行卡方值计算
        chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
        for i in np.arange(np_regroup.shape[0] - 1):
            chi,_ = self.chi2(np_regroup[i:i+2,1:3])
            chi_table = np.append(chi_table, chi)
        print('已完成变量{}数据初处理，正在进行卡方分箱核心操作'.format(variable))
        #把卡方值最小的两个区间进行合并（卡方分箱核心）
        while (1):
            if (len(chi_table) <= (bins - 1) or min(chi_table) >= confidenceVal):
                break
            chi_min_index = np.argmin(chi_table)# 找出卡方值最小的位置索引
            #print('最小索引%d' % chi_min_index)
            np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
            np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
            #np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
            #print(np_regroup)
#            print("np_regroup有{}行".format(np_regroup.shape[0]))
#            print("最小index:{}".format(chi_min_index))
#            print("cut_value最大index：{}".format(len(cut_value)-1))
            value_delete=cut_value.pop(chi_min_index+1)
            np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)
            #print("最小index:{}\n".format(chi_min_index))
            #print("此时的np_group{}".format(np_regroup))
            #print("此时的chi_table{}".format(chi_table))
            #print(np_regroup)
            if (chi_min_index == np_regroup.shape[0] - 1):  # 最小值试最后两个区间的时候
                # 计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1],error =self.chi2(np_regroup[chi_min_index-1:chi_min_index+1,1:3])
                # 删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index, axis=0)
            elif (chi_min_index == 0):
                chi_table[chi_min_index+1],error =self.chi2(np_regroup[chi_min_index:chi_min_index+2,1:3])
                # 删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index, axis=0)
            else:
                # 计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1],error = self.chi2(np_regroup[chi_min_index-1:chi_min_index+1,1:3])
                # 计算合并后当前区间与后一个区间的卡方值并替换
                chi_table[chi_min_index],error =self.chi2(np_regroup[chi_min_index:chi_min_index+2,1:3])
                # 删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
            if error==1:
                print("特征{}卡方分箱有误".format(variable))
        print('已完成变量{}卡方分箱核心操作，正在保存结果'.format(variable))
        self.cut_dict[variable]=cut_value
        return chi_table,cut_value
    def bins_gen(self,df,flag,break_dict={}):
        '''
        return : 变量切分字典映射表
        df : 数据集
        flag : 目标变量label,离散类型
        '''
        variables=[var for var in df.columns if var != flag]
        #df_new=pd.DataFrame()
        cut_dict={}
        assert(type(break_dict)==dict),"break_dict参数不是字典"
        #df=df.fillna('missing')
        assert(np.sum(df[flag].isna())==0),'目标变量存在缺失值'
        #df_not_missing=df[df!='missing']
        for var in variables:
            if is_numeric_dtype(df[var]):
                self.var_type[var]=1
            elif is_string_dtype(df[var]):
                self.var_type[var]=0
            else:
                raise Exception("变量勒边非数值非字符串类型",var)
            if var not in break_dict.keys():
                # #先用等深粗略切分成20分箱，在考虑合并
                #df_cut,cut_value=self.cut(df,var,20)
                #df_chi=pd.merge(df_cut,df[flag],left_index=True,right_index=True,how='inner')
                #df_chi.columns=[var,flag]
                if is_numeric_dtype(df[var]):
                    #self.var_type[var]=1
                    chi_value,cut_off=self.ChiMerge(df,var,flag)
                    var_cut=pd.cut(df[var],self.cut_dict[var])
                    var_cut=var_cut.cat.add_categories("missing").fillna("missing")
                    var_new=pd.concat([var_cut,df[flag]],axis=1)
                    #var_new.columns=[var,flag]
                    #woe_iv= self.woe_cal(var_new,var,flag)
                elif is_string_dtype(df[var]):
                    #self.var_type[var]=0
                    var_cut=df[var].fillna('missing')
                    var_new=pd.concat([var_cut,df[flag]],axis=1)
                    #woe_iv=self.woe_cal(var_new,var,flag)
                #cut_dict[var]=cut_off
                    self.cut_dict[var]=df[var].unique()
            else :
               # df_cut=pd.cut(df[var],break_dict[var])
                #cut_dict[var]=break_dict[var]
                self.cut_dict[var]=break_dict[var]
                var_cut=pd.cut(df[var],self.cut_dict[var])
                var_cut=var_cut.cat.add_categories("missing").fillna("missing")
                var_new=pd.concat([var_cut,df[flag]],axis=1)
            #var_new.columns=[var,flag]
            woe_iv= self.woe_cal(var_new,var,flag)
# =============================================================================
#             woe_iv.reset_index(inplace=True)
#             woe_iv.drop(var,axis=1,inplace=True)
# =============================================================================
            cut_dict[var]=woe_iv
        #self.cut_transform()
        return cut_dict
    def woe_cal(self,df,variable,flag):
        '''
        return : 单变量变换分箱后的值，变量各箱的WOE值,变量的IV值
        df : 数据集
        variable : 待切分变量/特征
        flag : 目标变量label,离散类型
        '''
		laplace=df[variable].nunique()  #拉普拉斯平滑系数
        total_bad=df[flag].sum()
        total_cnt=df[flag].count()
        total_good=total_cnt-total_bad
        group_bad=df.groupby([variable])[flag].sum()
        total=df.groupby([variable])[flag].count()
        group_good=total-group_bad
        bad_rate=group_bad/total
        group_rate=(group_bad+1)/(group_good+laplace)  #拉普拉斯平滑，防止出现某区间只有一类数据
        total_rate=(total_bad+1)/(total_good+laplace)  #拉普拉斯平滑，防止出现某区间只有一类数据
        woe=np.log(group_rate/total_rate)
        iv_bin=(group_bad/total_bad-group_good/total_good)*woe
        iv=np.sum(iv_bin)
        breaks=list(woe.index)
        if self.var_type[variable]:
            breaks=[c.right for c in breaks if c!='missing']
            breaks.append('missing')            
        else:
            pass            
#        print('切分点长度为{}'.format(len(breaks)))
#        print('其他长度为{}'.format(iv_bin.shape[0]))
#        print(breaks)
#        print(iv_bin)
        summary=pd.DataFrame({'feature':[variable]*iv_bin.shape[0],'cut':list(woe.index),'count':total,'distibution':round(total/total_cnt,4),
                              'good':group_good,'bad':group_bad,'bad_prob':bad_rate,'WOE':woe.values,'IV':iv_bin.values,'IV_total':[iv]*iv_bin.shape[0],
                              'breaks':breaks})

#        dictmap={}
#        for x in woe.index:
#            dictmap[x]=woe[x]
#        df_new=df[variable].map(lambda x : dictmap[x])
        summary.reset_index(inplace=True)
#        print(summary)
        summary.drop(variable,axis=1,inplace=True)
        return summary
    def woe_transform(self,df,cut_list):
        '''
        return : woe_data: woe变换后的数据集，woe_dict:，画出各变量IV值柱状图
        df : 数据集
        cut_list : 切分规则（字典形式）
        '''
        #cutdict = {}
        all_columns=list(df.columns)
        feature_columns=list(cut_list.keys())
        flag=list(set(all_columns).difference(set(feature_columns)))[0]
        ivdict = {}
        #woe_dict = {}
        woe_data = pd.DataFrame()
        df_bin=self.cut_transform(df,cut_list)
        #df_bin=df_bin.fillna('missing')
        for var in cut_list.keys():
            dictmap=cut_list[var].set_index('cut')['WOE'].to_dict()
            #if var != flag:
                #print(df.head(2))
                #print(var)
            variable_woe=df_bin[var].map(lambda x : dictmap[x])
#            woe_dict[var] = woe_iv['WOE'].tolist()
            woe_data[var] = variable_woe
            ivdict[var] = cut_list[var]['IV_total'].loc[0]
            #cutdict[var] = woe_iv['cut'].tolist()
                
        woe_data=woe_data.merge(df[flag],right_index=True,left_index=True,how='inner')
        ivdict = sorted(ivdict.items(), key=lambda x:x[1], reverse=False)
        iv_vs = pd.DataFrame([x[1] for x in ivdict],index=[x[0] for x in ivdict],columns=['IV'])
        ax = iv_vs.plot(kind='barh',
                        figsize=(12,12),
                        title='Feature IV',
                        fontsize=10,
                        width=0.8,
                        color='#00688B')
        ax.set_ylabel('Features')
        ax.set_xlabel('IV of Features')
        return woe_data
    def mono_test(self,cut_list):
        '''
        return : 特征是否单调,0表示不单调，1表示单调
        cut_list : 切分列表
        '''
        mono_var={}
        i=0
        for var in cut_list:
            woe_diff=np.array(cut_list[var]['WOE'].diff().dropna())
            if (all(woe_diff>=0) or all(woe_diff<=0)):
                mono_var[var]=1
            else:
                mono_var[var]=0
            i+=1
        mono_num=np.sum(list(mono_var.values()))
        print("共{}个变量woe值单调,有{}变量woe值不单调".format(mono_num,i-mono_num))
        return mono_var
        

        
        
        
if __name__=='__main__':
    from sklearn.model_selection import train_test_split 
    trainData = pd.read_csv('C:/Users/lancexiong/Downloads/A_Model_of_Risk_Control-master/A_Model_of_Risk_Control-master/test.csv')
    trainData.drop(['Unnamed: 0'],axis=1,inplace=True)
    train,test=train_test_split(trainData,test_size=0.3)
    #先等频分箱成20份
    bins=Binning()
    V_cut={'V01':[np.float('-inf'),740,750,755,761,np.float('inf')]}
    #chi_value,cut_value=bins.ChiMerge(d_v,'V01','result')
    #chi_value,cut_value=bins.ChiMerge(train,'V07','result')
    bins_gen=bins.bins_gen(train,'result',V_cut)
    #bins_gen=bins.bins_gen(train,'result')
    test_cut=bins.cut_transform(test,bins_gen)
    train_cut=bins.cut_transform(train,bins_gen)
    train_woe=bins.woe_transform(train,bins_gen)
	model_var=[i for i,j in bins_gen.item() if j['IV_total'].max()>=0.02]
    
    