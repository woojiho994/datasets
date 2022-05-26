# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:34:51 2021

@author: Jiho3
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pprint as pp
import joblib
import missingno
# import pdpbox
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['lines.linewidth'] = 2.2
plt.rcParams['axes.unicode_minus']=False 
#%%

#%% 调色盘
name_palette = sns.color_palette('tab20c_r',20)
sns.palplot(name_palette)
Nanoparticle_palette = sns.color_palette('tab20c',15)
current_palette = sns.color_palette('hls',15)
sns.palplot(current_palette)
#%%
filename = 'D:/我的坚果云/3.1-数据汇总(processed) (1).xlsx'
raw_data = pd.read_excel(filename,header=0,sheet_name='9.7+0数据')
raw_data.columns  #看特征名
data = raw_data[(raw_data['器官相加']==0)&(raw_data['净化']!=1)|(raw_data['器官相加']==3)]  #只选取非器官相加的数据
ft = ['Nanoparticle','Density(g/cm³)','Specific surface area(m²/g)','Surface modification',
      'Size(nm)','Hydrodynamic diameter(nm)','Zeta potential(mV)','name','Organism','DO','pH','Temperature(℃)','Shape',
      'Body length(mm)','Body weight(mg)','Life stage', 'NOM','Exposure time(h)','Exposure concentration(mg/L)','log10(BCF)','logBB']  
#元素、摩尔质量、密度、比表面积、尺寸、水动力学直径、zeta电位、生物种类、溶解氧、pH、温度、生物体长、暴露时间、暴露浓度
#未考虑因素：表面修饰s
data.drop(['备注', '1.11-检查环境浓度'],axis = 1,inplace=True)
df = data.loc[:,ft]
df = df.drop(df[(df['Nanoparticle'] =='Na2SeO3')|(df['Nanoparticle'] =='AgNO3')|(df['Nanoparticle'] == 'AuTiO2')|(df['Nanoparticle']=='CdS')].index)
#临时去掉一些数据
# df = df.drop(df[(df.Nanoparticle=='Cu')].index)
df = df.drop(df[(df['logBB']==-10)].index)  #去掉自定义0时刻的数据
# len(df['文献名字'].unique())

#%%
import missingno as msno
df1  = df.drop(columns=['log10(BCF)','name','logBB'])
msno.matrix(df1, labels=True,fontsize=28)#无效数据密度显示
 
# msno.bar(data)#条形图显示
 
# msno.heatmap(data)#热图相关性显示
 
# msno.dendrogram(data)#树状图显示
 

 
    #%%  取log和没取log的数据分布
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(16,5))
sns.histplot(df['BCF (L/kg)'],ax=ax[0])
sns.histplot(df['LOG10(BCF)'],ax=ax[1])
ax[0].set_xlabel('Pre Normalize\nSkew:%f \nKurt:%f '%(df['BCF (L/kg)'].skew(),df['BCF (L/kg)'].kurt()))
ax[1].set_xlabel('Post Normalize\nSkew:%f \nKurt:%f'%(df['LOG10(BCF)'].skew(),df['LOG10(BCF)'].kurt()))
plt.show()  
#%% 统计不同元素的数据数量
xtick = ['Nanoparticle','Organism']
tick = xtick[1]
sns.countplot(x=tick,data=df,palette = current_palette,order=df[tick].value_counts().index) 
plt.xticks(rotation=90,fontsize=12,family='Arial')
plt.ylabel('Count',fontsize=20,family = 'Arial')
plt.xlabel('Type of Nanoparticle',fontsize=20,family = 'Arial')
#plt.savefig('C:/Users/Lenovo/Desktop/Count.jpg', dpi=300,bbox_inches="tight")
print(df[tick].value_counts())
#%% 统计不同生物的数据数量
sns.countplot(x = 'Organism',data = df,palette=current_palette,order = df['Organism'].value_counts().index)
plt.xticks(rotation=90,fontsize=10,family='Arial')
plt.ylabel('Count',fontsize=20,family = 'Arial')
plt.xlabel('Type of Organisms',fontsize=20,family = 'Arial')
#%%  某种动物BCF的取值
data['LOG10(BCF)'][data.name=='斑马鱼'].describe()
#%%  不同动物BCF的取值
sns.factorplot(x = 'name', y = 'LOG10(BCF)',data = df,kind= 'bar')
plt.xticks(rotation=45)
plt.ylabel('logBCF',fontsize=20,family = 'Arial')
plt.xlabel('Type of Organisms',fontsize=20)
#%%  不同纳米材料BCF的取值
sns.factorplot(x = 'Nanoparticle', y = 'logBB',data = df,kind= 'bar')
plt.xticks(rotation=45,fontsize=10,family = 'Arial')
plt.ylabel('logBCF',fontsize=20,family = 'Arial')
plt.xlabel('Type of Nanoparticle',fontsize=20,family = 'Arial')
#%%
plt.figure(figsize=(6.5,4))
sns.stripplot(x='Organism',y='log10(BCF)',hue ='Nanoparticle',data=df,palette=Nanoparticle_palette)
ax = plt.gca()
bwith = 1.5
ax.spines['top'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,prop='Times New Roman')
plt.xticks(rotation=90,family = 'Times New Roman',fontsize=11,fontweight='bold')
plt.yticks(family='Arial',fontweight='bold')
plt.ylabel('log10(BCF)',fontsize=20,family = 'Times New Roman',fontweight='bold')
plt.xlabel('Type of Organisms',fontsize=20,family = 'Times New Roman',fontweight='bold')
# plt.savefig('C:/Users/Lenovo/Desktop/fix.jpg', dpi=300,bbox_inches="tight")
#%% 散点图，元素和BCF
plt.figure(figsize=(6.5,4))
sns.stripplot(x='Nanoparticle',y='LOG10(BCF)',hue ='name',data=df,palette=name_palette)
ax = plt.gca()
bwith = 1.5
ax.spines['top'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,prop = 'Times New Roman')
plt.xticks(rotation=90,fontsize=11,family = 'Times New Roman',fontweight='bold')
plt.ylabel('logBCF',fontsize=20,family = 'Times New Roman',fontweight='bold')
plt.xlabel('Type of Nanoparticles',fontsize=20,family = 'Times New Roman',fontweight='bold')
# plt.savefig('C:/Users/Lenovo/Desktop/fix.jpg',dpi=300,bbox_inches='tight')
#%%  材料-BCF箱式图
a = pd.DataFrame(columns=['logbb'])
for i in df.Nanoparticle.unique():
    a.loc[i] = df[df['Nanoparticle']==i]['logBB'].describe()['50%']
b = a.sort_values(by='logbb',ascending=False)
sns.boxplot(x='Nanoparticle',y='logBB',data=df,order=b.index)
ax = plt.gca()
bwith = 2
ax.spines['top'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.xlabel(None)
# plt.xlabel('Type of nanoparticles',fontsize=20,family = 'Times New Roman',fontweight='bold')
plt.ylabel('logBB',fontsize=20,family = 'Times New Roman',fontweight='bold')
plt.xticks(rotation=90,fontsize=14,family = 'Times New Roman',fontweight='bold')
plt.yticks(fontsize=14,family = 'Times New Roman',fontweight='bold')
# plt.savefig('D:/我的坚果云/材料-箱式图.jpg',bbox_inches='tight')
#%% 生物-箱式图
a = pd.DataFrame(columns=['logbb'])
for i in df.Organism.unique():
    a.loc[i] = df[df['Organism']==i]['logBB'].describe()['50%']
b = a.sort_values(by='logbb',ascending=False)
# plt.figure(figsize=(16,18))
sns.boxplot(x='Organism',y='logBB',data=df,order=b.index)
ax = plt.gca()
bwith = 2
ax.spines['top'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.xlabel('Type of organisms',fontsize=20,family = 'Times New Roman',fontweight='bold',labelpad=10)
plt.ylabel('logBB',fontsize=20,family = 'Times New Roman',fontweight='bold')
plt.xticks(rotation=90,fontsize=13,family = 'Times New Roman',style='italic',fontweight='bold')
plt.yticks(fontsize=14,family = 'Times New Roman',fontweight='bold')
# plt.savefig('D:/我的坚果云/小论文/图/生物-箱式图.jpg',bbox_inches='tight')
#%% 生物-箱式图(按生物体长排序)
# a = pd.DataFrame(columns=['logbcf'])
# for i in df.Organism.unique():
#     a.loc[i] = df[df['Organism']==i]['LOG10(BCF)'].describe()['50%']
# b = a.sort_values(by='logbcf',ascending=False)

b =  ['Daphnia magna','Capitella teleta','Artemia salina','Cyprinus carpio Larvae','Scrobicularia plana',
      'Limnodrilus hoffmeisteri','Danio rerio','Mytilus galloprovincialis','Nereis diversicolor']  #体长依次从小到大

sns.boxplot(x='Organism',y='LOG10(BCF)',data=df,order=b)
plt.xlabel('Type of organism',fontsize=20,family = 'Times New Roman',fontweight='bold',labelpad=10)
plt.ylabel('logBCF',fontsize=20,family = 'Times New Roman',fontweight='bold')
plt.xticks(rotation=90,fontsize=11,family = 'Times New Roman',style='italic',fontweight='bold')
plt.savefig('D:/我的坚果云/小论文/图/生物-箱式图.jpg',bbox_inches='tight')
#%% 生物-箱式图（中文名）
a = pd.DataFrame(columns=['logbcf'])
for i in df.name.unique():
    a.loc[i] = df[df['name']==i]['LOG10(BCF)'].describe()['50%']
c = a.sort_values(by='logbcf',ascending=False)
sns.boxplot(x='name',y='LOG10(BCF)',data=df,order=c.index)
plt.xlabel('Type of organism',fontsize=20,family = 'Times New Roman',fontweight='bold',labelpad=10)
plt.ylabel('logBCF',fontsize=20,family = 'Times New Roman',fontweight='bold')
plt.xticks(rotation=90,fontsize=11,family = 'SimHei',fontweight='bold')
plt.savefig('D:/我的坚果云/小论文/图/生物-箱式图(中文).jpg',bbox_inches='tight')

#%%   将金属和碳的分开
df['Nanoparticle'].unique()
C = ['C60','Graphene','MWCNTs']
carbon = df[df['Nanoparticle'].isin(C)]  #这个太强了
metal = df[~df['Nanoparticle'].isin(C)]
carbon.isnull().sum()
metal.isnull().sum()
#%% 将生物分成若干类别
df.name.unique()
fish_kind = ['斑马鱼', '罗非鱼', '金头鲷', '虹鳟鱼', '鲫鱼', '鲤鱼','黑鱼','日本青鳉']
fish = df[df['name'].isin(fish_kind)]
other = df[~df['name'].isin(fish_kind)]
#%% 10.8 大型蚤数据
magna = df[df['name']=='大型蚤'] 
#%% 鱼类统计
a = pd.DataFrame(columns=['logbcf'])
for i in fish.Organism.unique():
    a.loc[i] = fish[fish['Organism']==i]['LOG10(BCF)'].describe()['50%']
b = a.sort_values(by='logbcf',ascending=False)
sns.boxplot(x='Organism', y='LOG10(BCF)', hue=None, data=fish, order=b.index, hue_order=None, 
                orient=None, color=None, palette=None, saturation=0.75, width=0.8, dodge=True, fliersize=5, 
                linewidth=None, whis=1.5, notch=False, ax=None)
plt.xlabel('Type of organisms',fontsize=20,family='Times New Roman',fontweight='bold',labelpad=10)
plt.ylabel('logBCF',fontsize=20,family='Times New Roman',fontweight='bold',labelpad=10)
plt.xticks(rotation=90,fontsize=15,family='Times New Roman',fontweight='bold')
plt.yticks(fontsize=15,family='Times New Roman',fontweight='bold')
plt.savefig('D:/我的坚果云/fish.jpg',dpi=100,bbox_inches='tight')

#%%其他类生物统计
a = pd.DataFrame(columns=['logbcf'])
for i in other.Organism.unique():
    a.loc[i] = other[other['Organism']==i]['LOG10(BCF)'].describe()['50%']
b = a.sort_values(by='logbcf',ascending=False)
sns.boxplot(x='Organism', y='LOG10(BCF)', hue=None, data=other, order=b.index, hue_order=None, 
                orient=None, color=None, palette=None, saturation=0.75, width=0.8, dodge=True, fliersize=5, 
                linewidth=None, whis=1.5, notch=False, ax=None)
plt.xlabel('Type of organisms',fontsize=20,family='Times New Roman',fontweight='bold',labelpad=15)
plt.ylabel('logBCF',fontsize=20,family='Times New Roman',fontweight='bold',labelpad=10)
plt.xticks(rotation=90,fontsize=15,family='Times New Roman',fontweight='bold')
plt.yticks(fontsize=15,family='Times New Roman',fontweight='bold')
plt.savefig('D:/我的坚果云/other.jpg',dpi=100,bbox_inches='tight')

#%% 金属和碳分开画图
a = pd.DataFrame(columns=['logbcf'])
for i in metal['Nanoparticle'].unique():
    a.loc[i]=metal[metal['Nanoparticle']==i]['LOG10(BCF)'].describe()['50%']
b = a.sort_values(by='logbcf',ascending=False)

sns.boxplot(x='Nanoparticle', y='LOG10(BCF)', hue=None, data=metal, order=b.index, hue_order=None, 
                orient=None, color=None, palette=None, saturation=0.75, width=0.8, dodge=True, fliersize=5, 
                linewidth=None, whis=1.5, notch=False, ax=None)
plt.xlabel('Type of nanoparticles',fontsize=20,family='Times New Roman',fontweight='bold',labelpad=10)
plt.ylabel('logBCF',fontsize=20,family='Times New Roman',fontweight='bold',labelpad=10)
plt.xticks(rotation=90,fontsize=15,family='Times New Roman',fontweight='bold')
plt.yticks(fontsize=15,family='Times New Roman',fontweight='bold')
# plt.savefig('D:/我的坚果云/metal.jpg',dpi=100,bbox_inches='tight')

#想办法将顺序按从median的大到小

#%%形状
sns.boxplot(x='shape',y='LOG10(BCF)',hue=('Nanoparticle'),data=df)
#%%表面修饰
sns.boxplot(x='Surface modification',y='LOG10(BCF)',data=df)
plt.xticks(rotation=90,family='Microsoft YaHei')
# plt.savefig('D:/我的坚果云/6.24.jpg',dpi=100,bbox_inches='tight')
#%% 查看数据缺失情况
df.info()
df.isnull().sum()

#%% 填补数据(nanoparticle)
def fill_nan(data,feature):  #把‘/’都换成空的
    data[feature].replace('/',np.nan,inplace=True)


def fill_feature(data,feature):
    feature_list = list(df[df[feature].isnull()].Nanoparticle.unique())
    for i in feature_list:
        if df[feature][df.Nanoparticle==i].mean != np.nan:
            df[feature][df.Nanoparticle==i] = df[feature][df.Nanoparticle==i].fillna(df[feature][df.Nanoparticle==i].median())
        else:
            print(i)

def fill_median(data,feature):
    data[feature] = data[feature].fillna(data[feature].median())
#%%
def fill_namefeature(data,feature):
    feature_list = list(df[df[feature].isnull()].name.unique())
    for i in feature_list:
        if df[feature][df.name==i].mean != np.nan:
            df[feature][df.name==i] = df[feature][df.name==i].fillna(df[feature][df.name==i].median())
        else:
            print(i)

def fill_namemedian(data,feature):
    data[feature] = data[feature].fillna(data[feature].median()) 
    #%%填补shape
df['Shape'] = df['Shape'].fillna('spherical')
#填补表面修饰和NOM
df['Surface modification'] = df['Surface modification'].replace(np.nan ,'None')
df['Surface modification'] = df['Surface modification'].replace('/','None')
df['NOM'] = df['NOM'].replace(np.nan,'None')
#%%
# fill_namefeature(df,'bodylength(mm)')
# fill_namemedian(df,'bodylength(mm)')

fill_nan(df, 'Hydrodynamic diameter(nm)')
fill_feature(df,'Hydrodynamic diameter(nm)') 
fill_median(df, 'Hydrodynamic diameter(nm)')

fill_feature(df,'Specific surface area(m²/g)')
fill_median(df,'Specific surface area(m²/g)')

fill_feature(df,'Density(g/cm³)')
fill_median(df,'Density(g/cm³)')

fill_nan(df, 'Size(nm)')
fill_feature(df,'Size(nm)') 
fill_median(df, 'Size(nm)')
 
fill_nan(df, 'Zeta potential(mV)')
fill_feature(df,'Zeta potential(mV)') 
fill_median(df, 'Zeta potential(mV)')

# fill_median(df, 'molar mass')
#  DO pH temperature
fill_median(df,'DO')
fill_median(df,'pH')
fill_median(df,'Temperature(℃)')  #为什么温度影响大

#%%  再次查看
df.isnull().sum()
# df['Hydrodynamic diameter'].replace('/',np.nan,inplace=True)
# df['Hydrodynamic diameter'] = df['Hydrodynamic diameter'].astype(np.float32)
# sns.boxplot(y='Hydrodynamic diameter',x='Nanoparticle',data=df)
# df['Hydrodynamic diameter'].unique()

#%% 检查是否还有空值
df[ft].isnull().sum()
#%%  特征分布
sns.boxplot(x='Nanoparticle',y='Size',data=df)
plt.xticks(rotation=45,fontsize=10,family = 'Arial')
plt.ylabel('Size',fontsize=20,family = 'Arial')
plt.xlabel('Type of Nanoparticles',fontsize=20,family = 'Arial')
#%%
from sklearn.linear_model import LinearRegression
#%%  画图
h = sns.jointplot(x='Zeta potential(mV)',y='logBB',data=df,kind = 'reg',color='#286c43',ratio=5,space=0,height=6)
h.set_axis_labels('x', 'logBB(ug/g)', fontsize=25, family = 'Arial',fontweight='bold')
h.ax_joint.set_xlabel('Zeta Potential(mV)', fontweight='bold',fontsize=25)
plt.xticks(family = 'Arial',fontsize=16)
plt.yticks(family='Arial',fontsize=16)
h.ax_joint.set_xlim(-50,49)
plt.tight_layout()
# plt.savefig('D:/我的坚果云/小论文/11.1-更新图片/zeta.jpg',dpi=100,bbox_inches='tight')
plt.show()
df['Zeta potential(mV)'].copy()
#%%
h = sns.jointplot(x='Hydrodynamic diameter(nm)',y='logBB',data=df,kind = 'reg',color='#286c43',ratio=5,space=0,height=6)
h.set_axis_labels('x', 'logBB(ug/g)', fontsize=25, family = 'Arial', fontweight='bold')
h.ax_joint.set_xlabel('Hydrodynamic diameter(nm)', family = 'Arial', fontweight='bold',fontsize=25)
plt.xticks(family = 'Arial',fontsize=16)
plt.yticks(family='Arial',fontsize=16)
h.ax_joint.set_xlim(-2,2010)
plt.tight_layout()
# plt.savefig('D:/我的坚果云/小论文/11.1-更新图片/水动力学.jpg',dpi=100,bbox_inches='tight')
plt.show()
#%%
# df['Size'] = df['Size'].astype(np.float32)
h = sns.jointplot(x='Size(nm)',y='logBB',data=df,kind = 'reg',color='#286c43',ratio=5,space=0,height=6)
h.set_axis_labels('x', 'logBB(ug/g)',  fontsize=25, family = 'Arial', fontweight='bold')
h.ax_joint.set_xlabel('Size(nm)', family = 'Arial' ,fontweight='bold', fontsize=25)
plt.xticks(family = 'Arial',fontsize=16)
plt.yticks(family='Arial',fontsize=16)
h.ax_joint.set_xlim(-2,202)
plt.tight_layout()
# plt.savefig('D:/我的坚果云/小论文/11.1-更新图片/size.jpg',dpi=100,bbox_inches='tight')
plt.show()
#%%
h = sns.jointplot(x='Exposure time(h)',y='logBB',data=df,kind = 'reg',color='#286c43',ratio=5,space=0,height=6)
h.set_axis_labels('x', 'logBB(ug/g)', fontsize=25,family = 'Arial' ,fontweight='bold')
h.ax_joint.set_xlabel('Exposure time(h)',family = 'Arial' ,fontweight='bold',fontsize=25)
plt.xticks(family = 'Arial',fontsize=16)
plt.yticks(family='Arial',fontsize=16)
h.ax_joint.set_xlim(-2,121)
plt.tight_layout()
# plt.savefig('D:/我的坚果云/小论文/11.1-更新图片/暴露时间.jpg',dpi=100,bbox_inches='tight')
plt.show()
#%%
h = sns.jointplot(x='Exposure concentration(mg/L)',y='logBB',data=df,kind = 'reg',color='#286c43',ratio=5,space=0,height=6)
h.set_axis_labels('x', 'logBB(ug/g)',fontsize=25,family='Arial',fontweight='bold')
h.ax_joint.set_xlabel('Exposure concentration(mg/L)',family='Arial',fontweight='bold',fontsize=25)
plt.xticks(family = 'Arial',fontsize=16)
plt.yticks(family='Arial',fontsize=16)
h.ax_joint.set_xlim(-3,110)
plt.tight_layout()
# plt.savefig('D:/我的坚果云/小论文/11.1-更新图片/浓度.jpg',dpi=100,bbox_inches='tight')
plt.show()
#%%
h = sns.jointplot(x='Body length(mm)',y='logBB',data=df,kind = 'reg',color='#286c43')
h.set_axis_labels('x', 'logBB(ug/g)', fontsize=25,family='Arial',fontweight='bold')
h.ax_joint.set_xlabel('Body Length(mm)', family='Arial',fontweight='bold',fontsize=25)
plt.xticks(family = 'Arial',fontsize=16)
plt.yticks(family='Arial',fontsize=16)
h.ax_joint.set_xlim(-5,100)
plt.tight_layout()
# plt.savefig('D:/我的坚果云/小论文/11.1-更新图片/体长.jpg',dpi=100,bbox_inches='tight')
plt.show()
#%%
h = sns.jointplot(x='Specific surface area(m²/g)',y='logBB',data=df,kind = 'reg',color='#286c43')
h.set_axis_labels('x', 'logBB(ug/g)', fontsize=25,family='Arial',fontweight='bold')
h.ax_joint.set_xlabel('Specific surface area(m²/g)', family='Arial',fontweight='bold', fontsize=25)
plt.xticks(family = 'Arial',fontsize=16)
plt.yticks(family='Arial',fontsize=16)
h.ax_joint.set_xlim(0,130)
plt.tight_layout()
# plt.savefig('D:/我的坚果云/小论文/11.1-更新图片/体长.jpg',dpi=100,bbox_inches='tight')
plt.show()
#%%
h = sns.jointplot(x='Density(g/cm³)',y='logBB',data=df,kind = 'reg',color='#286c43')
h.set_axis_labels('x', 'logBB(ug/g)', fontsize=25,family='Arial',fontweight='bold')
h.ax_joint.set_xlabel('Density(g/cm³)', family='Arial',fontweight='bold',fontsize=25)
plt.xticks(family = 'Arial',fontsize=16)
plt.yticks(family='Arial',fontsize=16)
h.ax_joint.set_xlim(0,21)
plt.tight_layout()
# plt.savefig('D:/我的坚果云/小论文/11.1-更新图片/体长.jpg',dpi=100,bbox_inches='tight')
plt.show()
#%%
h = sns.jointplot(x='Temperature(℃)',y='logBB',data=df,kind = 'reg',color='#286c43')
h.set_axis_labels('x', 'logBB(ug/g)', fontsize=25,family='Arial',fontweight='bold')
h.ax_joint.set_xlabel('Temperature', family='Arial',fontweight='bold',fontsize=25)
plt.xticks(family = 'Arial',fontsize=16)
plt.yticks(family='Arial',fontsize=16)
h.ax_joint.set_xlim(8,31)
plt.tight_layout()
# plt.savefig('D:/我的坚果云/小论文/11.1-更新图片/体长.jpg',dpi=100,bbox_inches='tight')
plt.show()
#%%
df['Body weight(mg)'] = df['Body weight(mg)'].astype(float)
h = sns.jointplot(x='Body weight(mg)',y='logBB',data=df,kind = 'reg',color='#286c43')
h.set_axis_labels('x', 'logBB(ug/g)', fontsize=25,family='Arial',fontweight='bold')
h.ax_joint.set_xlabel('Body weight(mg)', family='Arial',fontweight='bold', fontsize=25)
plt.xticks(family = 'Arial',fontsize=16)
plt.yticks(family='Arial',fontsize=16)
h.ax_joint.set_xlim(0,13000)
plt.tight_layout()
# plt.savefig('D:/我的坚果云/小论文/11.1-更新图片/体长.jpg',dpi=100,bbox_inches='tight')
plt.show()
#%% 建模
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import warnings
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# import eli5
#%% 拆分数据
processed_data = df
processed_data = processed_data.drop(['name'],axis=1)
processed_data['Body weight(mg)'] = processed_data['Body weight(mg)'].astype(float)
# processed_data = processed_data.drop(['shape'],axis=1)
# df = df.drop(['Surface modification'],axis=1)

### 纳米材料分类#已弃用
# =============================================================================
# substance = ['Cu','SeNPs','ZnNPs','AgNPs','AuNPs']
# compound = ['ZnO','CeO2','CuO','TiO2','Ag2S','Fe3O4','CdS']
# C = ['C60','Graphene','MWCNTs']
# processed_data['substance'] = processed_data['Nanoparticle'].apply(lambda x: 1 if x in substance else 0)
# processed_data['compound'] = processed_data['Nanoparticle'].apply(lambda x: 1 if x in compound else 0)
# processed_data['C'] = processed_data['Nanoparticle'].apply(lambda x: 1 if x in C else 0)
# =============================================================================
###

###纳米材料分类
metal = ['CuNPs','SeNPs','ZnNPs','AgNPs','AuNPs','ZnO','CeO2','CuO','TiO2','Ag2S','Fe3O4','CdS']
C = ['C60','Graphene','MWCNTs']
processed_data['metal'] = processed_data['Nanoparticle'].apply(lambda x: 1 if x in metal else 0)
processed_data['C'] = processed_data['Nanoparticle'].apply(lambda x: 1 if x in C else 0)
### 

### 生物分类
vertebrate = ['Danio rerio','Cyprinus carpio Larvae']
invertebrate = ['Nereis diversicolor','Limnodrilus hoffmeisteri','Capitella teleta']
plankton = ['Artemia salina','Daphnia magna']
shell = ['Mytilus galloprovincialis','Scrobicularia plana']
processed_data['vertebrate'] = processed_data['Organism'].apply(lambda x: 1 if x in vertebrate else 0)
processed_data['invertebrate'] = processed_data['Organism'].apply(lambda x: 1 if x in invertebrate else 0)
processed_data['plankton'] = processed_data['Organism'].apply(lambda x: 1 if x in plankton else 0)
processed_data['shell'] = processed_data['Organism'].apply(lambda x: 1 if x in shell else 0)
###
processed_data = pd.get_dummies(processed_data,columns=['Life stage'])
processed_data = processed_data.drop(['log10(BCF)'],axis=1)
#去掉碳材料
# processed_data = processed_data[processed_data['C']!=1]
# 去掉短期暴露的
# processed_data = processed_data[(processed_data['Exposure time(h)']>=12) | (processed_data['Exposure time(h)']==0)]   
# 去掉TiO2
# processed_data = processed_data[processed_data['Nanoparticle']!='TiO2']                          
processed_data = shuffle(processed_data,random_state=42)


# X = X.drop(['Nanoparticle'],axis=1)
#y = processed_data['log10(BCF)']

processed_data['Size(nm)'] = processed_data['Size(nm)'].astype(np.float64)
# le = LabelEncoder()  #生物种类
# X['Organism'] = le.fit_transform(X['Organism']) #打标签
le1 = LabelEncoder() #表面修饰
processed_data['Surface modification'] = le1.fit_transform(processed_data['Surface modification'])
# le2 = LabelEncoder() #元素
# X['Nanoparticle'] = le2.fit_transform(X['Nanoparticle'])

le3 = LabelEncoder() #形状
processed_data['Shape'] = le3.fit_transform(processed_data['Shape'])
le4 = LabelEncoder() # NOM
processed_data['NOM'] = le4.fit_transform(processed_data['NOM'])

# final_data = processed_data[processed_data['器官相加']==0]
X = processed_data.drop(['logBB'],axis=1)
X.drop(['Nanoparticle','Organism'],axis=1,inplace=True) 
y = processed_data['logBB']
n = processed_data.describe()
# test_data = processed_data[processed_data['器官相加']==3]
# test_x = test_data.drop(['logBB','器官相加'],axis=1)
# test_x.drop(['Nanoparticle','Organism'],axis=1,inplace=True) 
# test_y = test_data['logBB'] 


# joblib.dump(le,'生物打标签.pkl')
# X['Nanoparticle'] = le.fit_transform(X['Nanoparticle']) #打标签
# X = pd.get_dummies(X,columns=['Nanoparticle','name'])
#X = X.values
#y_logbcf = y_logbcf.values
#y_logbb = y_logbb.values
# =============================================================================
# 数据标准化
# std = StandardScaler()
# X_std = std.fit_transform(X)
# =============================================================================

#%% 拆分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))


#%% 画图的函数
def plot_regression(model,train,test,title):
    y_pred = model.predict(test)
    y_pred_train = model.predict(train)
    n= np.linspace(0.5,5)
    plt.figure(figsize=(6.5,6))
    plt.scatter(y_test,y_pred,c='b',alpha=0.7,label='test')
    plt.scatter(y_train,y_pred_train,c='r',alpha=0.1,label='train')
    plt.plot(n,n,'b',lw=1.5,alpha=0.5)
    plt.axis([-0.5,6,-0.5,6])
    #t = r'$R^2=0.79$'
    t = r'$R^2=%.2f$'%r2_score(y_pred,y_test)
    t1 = '$y=x$'
    rmse = r'RMSE=%.2f'%np.sqrt(mean_squared_error(y_test, y_pred))
    plt.text(4,1,t,fontsize=19,family = 'Arial')
    plt.text(4.2,5.25,t1,fontsize=20,family = 'Arial')
    plt.text(4,0.5,rmse,fontsize=19,family = 'Arial')
    plt.ylabel('Predicted',fontsize=17,family = 'Arial')
    plt.xlabel('Observed',fontsize=17,family = 'Arial')
    plt.title(title,fontsize=22,family = 'Arial')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.legend(loc='upper left',frameon=True,fontsize=15)
    plt.show()
#%% 为每个模型输出预测值,用于graph画图
def pred(model,train,test):
    y_pred_train = model.predict(train)
    y_pred_test = model.predict(test)
    return y_pred_train,y_pred_test
    
#%%   得把数据打乱再来交叉验证
random_state=42
Regressor = [DecisionTreeRegressor(random_state=random_state),
           SVR(),
           RandomForestRegressor(random_state=random_state),
           XGBRegressor(random_state=random_state)]
dt_param_grid={"min_samples_split": range(10,500,20),
              "max_depth": range(1,20,2)}
svr_param_grid={"kernel":["rbf"],
               "gamma":[0.001,0.01,0.1,1],#"C":[1,10,50,100,200,300,1000],#"probability" : [True]}
               }
rf_param_grid={"max_features":[1,3,5,10],
               #[1,3,10,],
              "bootstrap":[False],
              "max_depth":range(1,20,2),
              "n_estimators":range(50,500,50)
              }
xgb_params_grid={"learning_rate":[0.001,0.01,0.05,0.1],
                 "n_estimators":range(50,500,50),
                 "max_depth":range(1,8,1)}
Regressor_param=[dt_param_grid,svr_param_grid,rf_param_grid,xgb_params_grid]

cv_result=[]

best_estimators=[]

for i in range(len(Regressor)):
    model=GridSearchCV(Regressor[i],param_grid=Regressor_param[i],
                     cv=10,scoring = 'r2',#StratifiedKFold(n_splits=10),
                    n_jobs=-1,# for parallel running for higher speed
                    verbose=1  #show the results while the code is running
                    )
    model.fit(X,y)
    cv_result.append(model.best_score_)
    best_estimators.append(model.best_estimator_)
    print(cv_result[i])
    
#%% 画图对比多个模型结果
print(best_estimators)
#%% 画图
print(cv_result)
est = ['DecisionTree','Support Vector Machine','RandomForest','XGBoost']
rgb = list(['tomato','mediumseagreen','royalblue','darkorange','red'])
plt.bar(est,cv_result,color=rgb)    
plt.xticks(rotation=15,family='Arial',fontsize=12,fontweight='bold')    
plt.yticks(family='Arial',fontsize=15,fontweight='bold') 
plt.ylabel('Accuracy', fontsize=15)

 #%%xgb
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))

xgb = XGBRegressor(max_depth=15, learning_rate=0.01, n_estimators=500,booster = 'gbtree')
xgb.fit(X_train,y_train)
plot_regression(xgb, X_train, X_test,'XGBoostRegressor')
y_predtrain,y_predtest = pred(xgb,X_train,X_test)
print(y_predtrain)
print(y_predtest)

# scores = cross_val_score(xgb,X,y, cv=5)  #10折交叉验证
# print(scores)
# print(scores.mean())
#%% 3.10-分离部分数据作为外部测试集
print('r2:',r2_score(test_y,xgb.predict(test_x)))
print('Y：',np.round(test_y.values,3))
print('x：',np.round(xgb.predict(test_x),3))
#%%  3.10- 置换检验过拟合
xgb = XGBRegressor(max_depth=15, learning_rate=0.01, n_estimators=500,booster = 'gbtree')
num = 577  #随机取值的y的数量 116,232,347,463
r = np.random.uniform(-1.79265,5.12388,num)
y_per = y.copy()
y_per[:num] = r
print('随机y和原本y的相关系数：',np.round(y.corr(y_per,method='pearson'),3))
scores = cross_val_score(xgb,X,y_per, cv=10,scoring='r2')
print('Q2:',scores)
#%% adaboost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test)) 
 
ada = AdaBoostRegressor(DecisionTreeRegressor())
ada.fit(X_train,y_train)
plot_regression(ada, X_train, X_test,'AdaBoostRegressor')
y_predtrain,y_predtest = pred(ada,X_train,X_test)
print(y_predtrain)
print(y_predtest)
# scores = cross_val_score(ada,X,y, cv=10)  #10折交叉验证
# print(scores)
# print(scores.mean())
#%%决策树
DT = DecisionTreeRegressor()
DT.fit(X_train,y_train)
plot_regression(DT, X_train, X_test,'DecisionTreeRegressor')
y_predtrain,y_predtest = pred(DT,X_train,X_test)
print(y_predtrain)
print(y_predtest)
# scores = cross_val_score(DT,X,y, cv=10)  #10折交叉验证
# print(scores)
# print(scores.mean())
#%% svr
svr = SVR(gamma=0.0001)
svr.fit(X_train,y_train)
plot_regression(svr, X_train, X_test,'SVMRegressor')
y_predtrain,y_predtest = pred(svr,X_train,X_test)
print(y_predtrain)
print(y_predtest)
scores = cross_val_score(svr,X,y, cv=10)  #10折交叉验证
print(scores)
print(scores.mean())
#%% linear
linreg = LinearRegression()
linreg.fit(X_train,y_train)
plot_regression(linreg, X_train, X_test,'LinearRegressor')
y_predtrain,y_predtest = pred(linreg,X_train,X_test)
print(y_predtrain)
print(y_predtest)
scores = cross_val_score(linreg,X,y, cv=10)  #10折交叉验证
print(scores)
print(scores.mean())
#%% 决策树
DT = DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42)
DT.fit(X_train,y_train)
plot_regression(DT, X_train, X_test,'DecisionTreeRegressor')
score = DT.score(X_test,y_test)
print(score)
scores = cross_val_score(DT,X,y, cv=10)  #10折交叉验证
print(scores)
print(scores.mean())
#%% 神经网络
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))

nn = MLPRegressor(solver='lbfgs',activation='tanh',
                  hidden_layer_sizes=(5,20,20,20),max_iter=10000,alpha=1e-4)
nn.fit(X_train,y_train)
plot_regression(nn, X_train, X_test,'MLPRegressor')
y_predtrain,y_predtest = pred(nn,X_train,X_test)
print(y_predtrain)
print(y_predtest)
# scores = cross_val_score(nn,X,y, cv=10)  #10折交叉验证
# print(scores)
# print(scores.mean())

#%% 岭回归
from sklearn.linear_model import Ridge
rgr = Ridge(alpha=0.001)
rgr.fit(X_train,y_train)
plot_regression(rgr,X_train, X_test,'RidgeRegressor')
y_predtrain,y_predtest = pred(rgr,X_train,X_test)
print(y_predtrain)
print(y_predtest)

#%% lasso回归
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.0001)
ls.fit(X_train,y_train)
plot_regression(ls,X_train, X_test,'LassoRegressor')
y_predtrain,y_predtest = pred(ls,X_train,X_test)
print(y_predtrain)
print(y_predtest)
# scores = cross_val_score(ls,X,y, cv=10)  #10折交叉验证
# print(scores)
# print(scores.mean())
#%% 拆分训练集和验证集——   随机森林
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))

rf = RandomForestRegressor(bootstrap=False, max_depth=8,max_features=20,n_estimators=500,criterion='mse')
rf.fit(X_train,y_train)
plot_regression(rf, X_train, X_test,'RandomForestRegressor')
y_predtrain,y_predtest = pred(rf,X_train,X_test)
print(y_predtrain)
print(y_predtest)
pp.pprint(rf.get_params())
# print('测试集R2=',rf.score(X_test,y_test))
# print('测试集RMSE=',np.sqrt(mean_squared_error(y_test, y_predtest)))
# print('训练集R2=',rf.score(X_train,y_train))
# =============================================================================
# scores = cross_val_score(rf,X,y, cv=10)  #10折交叉验证
# print(scores)
# print(scores.mean())
# =============================================================================
# joblib.dump(rf,'7.6-RF')
# joblib.dump(rf,'randomforest.pkl')
# rf2 = joblib.load('randomforest.pkl')
#%%
##  SHAP
# from IPython.display import (display, display_html, display_png, display_svg)
import shap
# shap.initjs()
explainer = shap.Explainer(xgb)
shap_values = explainer(X_train)

# shap.summary_plot(shap_values, X_train)
shap.plots.waterfall(shap_values[9])   ## 单个样本的特征贡献值
# print(xgb.predict(X_train).mean())
#%%
shap.plots.force(shap_values[0])

# visualize the first prediction's explanation with a force plot
shap.plots.force(shap_values[0])
plt.show()
# shap.summary_plot(shap_values[0], X_train)

shap.plots.force(shap_values)
shap.plots.force(explainer.expected_value,shap_values.values[:100],shap_values.data[:100])
#%%
explainer = shap.TreeExplainer(rf)
shap_value_single = explainer.shap_values(X = X_train.iloc[1,:])
shap.force_plot(base_value = explainer.expected_value[1],shap_values = shap_value_single[1],features = X_train.iloc[1,:])
#%%%
shap.plots.scatter(shap_values[:,"Body length(mm)"], color=shap_values[:,"Exposure concentration(mg/L)"])
#%%
shap.plots.bar(shap_values)
shap.plots.heatmap(shap_values)
#%%
clust = shap.utils.hclust(X, y, linkage="single")
shap.plots.bar(shap_values, clustering=clust, clustering_cutoff=1)
#%% 8.1绘制ICE图
from sklearn.inspection import plot_partial_dependence,partial_dependence

ice_feature = ['Density(g/cm³)']
#['Exposure time(h)',                

                # 'Body length(mm)','Size',
               #'Zeta potential','Hydrodynamic diameter','Density(g/cm³)#]
               
# feature_name = ['Exposure time(h)','Exposure concentration(mg/L)','bodylength(mm)']
plot_partial_dependence(xgb,X_train,features=ice_feature,kind='both')
plt.legend(loc='upper right',frameon=True,fontsize=12)
# plt.xlim(0,35)
# plt.ylim(0,5)
# plt.ylabel()
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
# plt.savefig('D:/我的坚果云/ice-暴露时间.jpg',dpi=300,bbox_inches='tight')
# fig.subplots_adjust(hspace=0.5)
# plot_partial_dependence(rf,X_test,ice_feature,kind='both')
#%% 手动导出pdp数据
pdp_y = partial_dependence(xgb,X,features=['Exposure time(h)'],kind='individual'
                                 # percentiles=[0,1],
                                 # grid_resolution=10  # 点的数量
                               ).individual
pdp_x = partial_dependence(xgb,X,features=['Exposure time(h)'],kind='individual'
                                 # percentiles=[0,1],
                                 # grid_resolution=10  # 点的数量
                               )['values']
# pdp_x[0]
# plt.plot(pdp_x[0],pdp_y)
#%%
## 2月8日 pdpbox图片
from pdpbox import pdp,info_plots
base_features = X.columns.values.tolist()
feature_name = ['Exposure time(h)','Exposure concentration(mg/L)','Density(g/cm³)',
                'Size(nm)','Body length(mm)','Surface area(m²/g)']
feat_name = feature_name[4]  
pdp_dist = pdp.pdp_isolate(
    model=xgb,  # 模型
    dataset=X_test,  # 测试集
    model_features=base_features,  # 特征变量；除去目标值 
    feature=feat_name  # 指定单个字段
)

pdp.pdp_plot(pdp_dist, feat_name,frac_to_plot=0.5, plot_lines=False,
              x_quantile=True, show_percentile=False, plot_pts_dist=True)  # 传入两个参数
plt.show()
#%%
## 对onehot变量进行分析
fig, axes, summary_df = info_plots.target_plot(
    df=processed_data, feature=['shell','plankton','invertebrate','vertebrate'],
    feature_name=['Organism type'],target='logBB'  # 这只是个列名字
)
#%%
fig, axes, summary_df = info_plots.target_plot(
    df=processed_data, feature=['Life stage_adult',
                        'Life stage_juvenile', 'Life stage_neonate'],
    feature_name=['life_stage'],target='logBB'  # 这只是个列名字
)
#%%  两个变量
fig, axes, summary_df = info_plots.target_plot(
    df=processed_data, feature=['Organism',['metal','C']],
    feature_name=['nano','Nanoparticle type'],target='logBB'  # 这只是个列名字
)
#%%
#%%
fig, axes, summary_df = info_plots.target_plot_interact(
    df=processed_data, feature=['metal','C']],
    feature_name=['Nanoparticle type'],target='logBB'  # 这只是个列名字
)
#%%
## 对onehot变量进行预测分析
fig, axes, summary_df = info_plots.actual_plot(
    model=xgb, X=X, 
    feature=['shell','plankton','invertebrate','vertebrate'], feature_name='Organism type'
)
print(summary_df)
#%%
## 对onehot变量进行预测分析
fig, axes, summary_df = info_plots.actual_plot(
    model=xgb, X=X, 
    feature=['metal','C'], feature_name='Nanoparticle type'
)
print(summary_df)
#%%
##  2d-pdp
inter1  =  pdp.pdp_interact(
    model=xgb,  # 模型
    dataset=X_test,  # 特征数据集
    model_features=base_features,  # 特征
    features=[feature_name[2], feature_name[4]])

pdp.pdp_interact_plot(
    pdp_interact_out=inter1, 
    feature_names=[feature_name[2], feature_name[4]], 
    plot_type='contour')
plt.show()
#%%  

from sklearn.inspection import permutation_importance
result = permutation_importance(rf, X, y, n_repeats=10, random_state=0,
                                n_jobs=-1)
 
fig, ax = plt.subplots()
sorted_idx = result.importances_mean.argsort()
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=range(X.shape[1]))
ax.set_title("Permutation Importance of each feature")
ax.set_ylabel("Features")
fig.tight_layout()
plt.show()
#%%
eli5.show_weights(rf, feature_names = list(X_test.columns))
html_obj = eli5.show_weights(rf, feature_names = list(X_test.columns))

# Write html object to a file (adjust file path; Windows path is used here)

with open('C:\\Users\\Lenovo\\Desktop\\iris-importance.htm','wb') as f:

    f.write(html_obj.data.encode("UTF-8"))  

# Open the stored HTML file on the default browser

url = r'C:\\Tmp\\Desktop\iris-importance.htm'


#%% 随机森林特征重要性+画图
# 特征名字
importances = list(rf.feature_importances_)
# 名字，数值组合在一起
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(X.columns, importances)]
# 排序  
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# 打印出来 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.figure(figsize=(10,6))
y_ = range(len(X_train.columns))
c = []  #特征重要性
n = []  #重要性值
for i in y_:
    c.append(feature_importances[i][0])
    n.append(feature_importances[i][1])    #用循环把X.column和重要性提出来
plt.bar(x=0,bottom=y_,height =0.8,align='edge',width=n,color='orangered')
plt.yticks(np.arange(len(X.columns)),c,horizontalalignment='right',verticalalignment='bottom')
plt.xlabel('importance',fontsize=20)
plt.ylabel('feature',fontsize=20)
plt.title('feature importance')
plt.show()
#%%  预测指定数据的BCF
#  AgNPs-大型蚤
#Bioaccumulation of silver in Daphnia magna:Waterborne and dietary exposure to nanoparticles and dissolved silver body_burden = pd.DataFrame()
body_burden = pd.DataFrame()

new_data = processed_data.loc[(processed_data['Nanoparticle']=='AgNPs')&(processed_data['Organism']=='Daphnia magna')
                              &(processed_data['Exposure concentration(mg/L)']==0.005)&(processed_data['Exposure time(h)']==48)]  #更改浓度获得不同数据

new_data['Exposure time(h)'] = 48
new_data['Surface modification'] = le1.transform(new_data['Surface modification'])
new_data['vertebrate'] = new_data['Organism'].apply(lambda x: 1 if x in vertebrate else 0)
new_data['invertebrate'] = new_data['Organism'].apply(lambda x: 1 if x in invertebrate else 0)
new_data['plankton'] = new_data['Organism'].apply(lambda x: 1 if x in plankton else 0)
new_data['shell'] = new_data['Organism'].apply(lambda x: 1 if x in shell else 0)
new_data['metal'] = new_data['Nanoparticle'].apply(lambda x: 1 if x in metal else 0)
new_data['C'] = new_data['Nanoparticle'].apply(lambda x: 1 if x in C else 0) 
new_data['shape'] = le3.transform(new_data['shape'])
new_data['NOM'] = le4.transform(new_data['NOM'])
x = new_data.drop(['log10(BCF)','Nanoparticle','Organism','logBB'],axis=1)
bcf = rf.predict(x)
print(bcf)
#%%  随机森林特征重要性_第二种方法
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = X_train.columns
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

#%% 画图(画不出来)
import graphviz
import pydotplus
import pydot
from sklearn import tree 
rf.estimators_[0]
with open("tree.dot", 'w',encoding='gbk') as f:
    f = tree.export_graphviz(rf.estimators_[0], out_file=None) 
    graph = pydotplus.graph_from_dot_data(f) 
    graph.write_png("tree.png") 

#%% 特征数量对结果的影响
# import9 = ['Organism','Exposure concentration(mg/L)','bodylength(mm)','Exposure time(h)','Size','Zeta potential'
#            ,'temperature','Hydrodynamic diameter','Nanoparticle']         
# non9 = ['molar mass','pH','C','Surface modification','shape','substance','compound'
#         ,'NOM','DO']
# feat = import9 + non9
r2 = []
rmse = []
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X[c[:-18]], y, test_size=0.3)
    rf_1 = RandomForestRegressor(bootstrap=False, max_depth=3,max_features=3,n_estimators=500)
    rf_1.fit(X_train,y_train)
    r2.append(rf_1.score(X_test,y_test))
    y_pred = rf_1.predict(X_test)
    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
print(r2,)
print(rmse)
plot_regression(rf_1, X_train, X_test,'RandomForestRegressor')
y_predtrain,y_predtest = pred(rf_1,X_train,X_test)
# print(y_predtrain)
# print(y_predtest)
#%% 随机森林特征重要性
# 特征名字
importances = list(rf.feature_importances_)

# 名字，数值组合在一起
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(X[feat[4:]].columns, importances)]

# 排序  
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# 打印出来 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#%% 单独筛选随机森林的最优参数
model = RandomForestRegressor()
rf_param_grid={"max_features":[1,3,5,10,12],#[1,3,10,],
              "bootstrap":[False],
              "n_estimators":[100,300,500,1000],
              }
model1 = GridSearchCV(model,param_grid=rf_param_grid,n_jobs=-1,cv=5,scoring='r2',verbose=1)
model1.fit(X,y)
print(model1.best_score_)
print(model1.best_estimator_)
# print(model1.best_score_)
# print(model1.best_estimator_)
# help(RandomForestRegressor())

#%%

# probs = model.predict_proba(X)

# probs_df = pd.DataFrame(probs, columns=['', ''])

# probs_df['was_correct'] = rf.predict(X) == y

# f, ax = plt.subplots(figsize=(7, 5))

# probs_df.groupby('').was_correct.mean().plot(kind='bar', ax=ax)

# ax.set_title("Accuracy at 0 class probability")

# ax.set_ylabel("% Correct")

# ax.set_xlabel("% trees for 0")

# f.show()


#%% 聚类
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=60,min_samples=3)
db.fit(X)
labels = db.labels_
print(labels)
X['cluster_db'] = labels
X.sort_values('cluster_db')
print(X.groupby('cluster_db').mean())    
#%%
# 画出在不同两个指标下样本的分布情况
ft2=['name','Nanoparticle','Size','Hydrodynamic diameter','Zeta potential','name',
      'body length', 'Equilibrium time(d)','Exposure concentration','cluster_db']
X2 = X[ft2]
pd.plotting.scatter_matrix(X, c=X.cluster_db,figsize=(10,10), s=100)

#%%
df1['cluster_db'] = labels
sns.boxplot(x = 'cluster_db',y = 'LOG10(BCF)',data=df1)
sns.boxplot(x = 'cluster_db',y = 'name',data=df1)
sns.boxplot(x = 'cluster_db',y = 'Nanoparticle',data=df1)

#%%画热图
# correlations / mask
corr = df[ft].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

# plot / cmap
fig, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .6})

plt.show()

#%%
data1 = df[ft]
data.isnull().sum()
# obtain missing features
dict_missing = {}   #将缺失值传入字典
for b in [a for a in data.columns if (data[a].isnull().sum() > 0) & (not a == 'logBCF')]: 
    dict_missing[b] = data[b].isnull().sum()
dict_missing = dict(sorted(dict_missing.items(), key=lambda item: item[1])) # sort by count
#%%  用knn填补缺失值
# enum each missing feature
data['Hydrodynamic diameter'].replace('/',np.nan,inplace=True) 
data['Zeta potential'].replace('/',np.nan,inplace=True) 
data['Hydrodynamic diameter']=data['Hydrodynamic diameter'].astype(np.float64)
for f in dict_missing:
    
    # 构造分类器
    missing_estimator = KNeighborsClassifier(n_neighbors=10) # default as classifier
    if (data[f].dtype == np.float64): 
        missing_estimator = KNeighborsRegressor(n_neighbors=10) # if float use regressor
    
    # 非空的行
    n_cols = [n for n in data.select_dtypes(include=['number']).columns if (data[n].isnull().sum()== 0) & (n not in ['LOG10(BCF)'])]
    
    # 拟合
    missing_estimator.fit(
        data[n_cols][(data[f].notnull())], # 所有非空数据/f也是非空
        data[f][(data[f].notnull())] # f是非空的
    )
    
    # 预测，填空
    data[f] = data.apply(lambda x:
                         missing_estimator.predict(np.array(x[n_cols]).reshape(1,-1))[0] if 
                         pd.isnull(x[f]) else x[f], axis=1) # predict missing



#%% 只看大型蚤
magna = df[df['name']=='大型蚤']
sns.stripplot(x = 'Nanoparticle', y = 'LOG10(BCF)',data = magna)
ax = plt.gca()
bwith = 1.5
ax.spines['top'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xticks(rotation=90,family = 'Times New Roman',fontsize=11,fontweight='bold')
plt.yticks(family='Arial',fontweight='bold')
plt.ylabel('logBCF',fontsize=20,family = 'Times New Roman',fontweight='bold',loc='bottom')
plt.xlabel('Type of Nanoparticles',fontsize=20,family = 'Times New Roman',fontweight='bold')
# plt.savefig('C:/Users/Lenovo/Desktop/fix.jpg', dpi=300,bbox_inches="tight")
#%% metal 建模
metal = shuffle(metal)
metal.drop(columns=['name'],axis=1,inplace=True)
X = metal.drop(['LOG10(BCF)'],axis=1)
y = metal['LOG10(BCF)']
X['Size'] = X['Size'].astype(np.float64)
le = LabelEncoder()  #打标签
X['Organism'] = le.fit_transform(X['Organism']) #打标签
X['Nanoparticle'] = le.fit_transform(X['Nanoparticle']) #打标签
# X = pd.get_dummies(X,columns=['Nanoparticle','name'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
#%%   得把数据打乱再来交叉验证
random_state=42
Regressor = [DecisionTreeRegressor(random_state=random_state),
           SVR(),
           RandomForestRegressor(random_state=random_state),
           XGBRegressor(random_state=random_state)]
dt_param_grid={"min_samples_split": range(10,500,20),
              "max_depth": range(1,20,2)}
svr_param_grid={"kernel":["rbf"],
               "gamma":[0.001,0.01,0.1,1],#"C":[1,10,50,100,200,300,1000],#"probability" : [True]}
               }
rf_param_grid={"max_features":[1,3,5,10],
               #[1,3,10,],
              "bootstrap":[False],
              "max_depth":range(1,20,2),
              "n_estimators":range(50,500,50),
              }
xgb_params_grid={"learning_rate":[0.001,0.01,0.05,0.1],
                 "n_estimators":range(50,500,50),
                 "max_depth":range(1,8,1)}
Regressor_param=[dt_param_grid,svr_param_grid,rf_param_grid,xgb_params_grid]

cv_result=[]

best_estimators=[]

for i in range(len(Regressor)):
    model=GridSearchCV(Regressor[i],param_grid=Regressor_param[i],
                     cv=10,scoring = 'r2',#StratifiedKFold(n_splits=10),
                    n_jobs=-1,# for parallel running for higher speed
                    verbose=1  #show the results while the code is running
                    )
    model.fit(X,y)
    cv_result.append(model.best_score_)
    best_estimators.append(model.best_estimator_)
    print(cv_result[i])
#%% 画图
print(cv_result)
est = ['DecisionTree','Support Vector Machine','RandomForest','XGBoost']
rgb = list(['tomato','mediumseagreen','royalblue','darkorange','red'])
plt.bar(est,cv_result,color=rgb)    
plt.xticks(rotation=15,family='Arial',fontsize=12,fontweight='bold')    
plt.yticks(family='Arial',fontsize=15,fontweight='bold') 
plt.ylabel('Accuracy', fontsize=15)
#%%
rf = RandomForestRegressor(bootstrap=False, max_depth=15, max_features=10,
                       n_estimators=200, random_state=42)
rf.fit(X_train,y_train)
plot_regression(rf, X_train, X_test,'RandomForestRegressor')
y_predtrain,y_predtest = pred(rf,X_train,X_test)
print(y_predtrain)
print(y_predtest)
pp.pprint(rf.get_params())