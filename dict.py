# %%

import pandas as pd
import json
import re


df_SOC = pd.read_excel( "./dictionaries/20220706 - MedDRA 25.0.xlsx", sheet_name="SOC",header=0)
df_HLGT = pd.read_excel( "./dictionaries/20220706 - MedDRA 25.0.xlsx", sheet_name="HLGT",header=0)
df_HLT = pd.read_excel( "./dictionaries/20220706 - MedDRA 25.0.xlsx", sheet_name="HLT",header=0)
df_PT = pd.read_excel( "./dictionaries/20220706 - MedDRA 25.0.xlsx", sheet_name="PT",header=0)
df_LLT = pd.read_excel( "./dictionaries/20220706 - MedDRA 25.0.xlsx", sheet_name="LLT",header=0)
df_compositions = pd.read_excel("./dictionaries/databases.xlsx", sheet_name="liste des compositions",header=None)
df_generiques = pd.read_excel("./dictionaries/databases.xlsx", sheet_name="Liste des génériques",header=None)
df_spécialités = pd.read_excel("./dictionaries/databases.xlsx", sheet_name="Liste des spécialités",header=None)

# %%

pattern = open("pattern.jsonl","w")

# add SOC to dictionary
# SOC = df_SOC["soc_name_fr"].to_list()
# for i in SOC:
#     line = {"label": "NLD","pattern": i }
#     json.dump(line, pattern)
#     pattern.write("\n")

# # add HGLT to dictionary
# HLGT = df_HLGT["hlgt_name_fr"].to_list()
# for i in HLGT:
#     line = {"label": "NLD","pattern": i }
#     json.dump(line, pattern)
#     pattern.write("\n")

# # add HLT to dictionary
# HLT = df_HLT["hlt_name_fr"].to_list()
# for i in HLT:
#     line = {"label": "NLD","pattern": i }
#     json.dump(line, pattern)
#     pattern.write("\n")

# # add PT to dictionary
# PT = df_PT["pt_name_fr"].to_list()
# for i in PT:
#     line = {"label": "NLD","pattern": i }
#     json.dump(line, pattern)
#     pattern.write("\n")

# # add LLT to dictionary
# LLT = df_LLT["llt_name_fr"].to_list()
# for i in SOC:
#     line = {"label": "NLD","pattern": i }
#     json.dump(line, pattern)
#     pattern.write("\n")


# add compositions to dictionary

#print(df_compositions.shape)
df_compositions.drop_duplicates()
#print(df_compositions.shape)
compositions = df_compositions.iloc[:,0].to_list()
for i in compositions:
    # extract only the words in uppercase letter, which are the name of our interest
    regex = re.compile(r'[A-ZÀ-ÖØ-Þ]+')
    words = regex.findall(i)
    line = {"label": "DRUG", "pattern": [{"LOWER": j}for j in words]}
    json.dump(line, pattern)
    pattern.write("\n")

# add generiques to dictionary


print(df_generiques.shape)
df_generiques.drop_duplicates()
print(df_generiques.shape)

generiques = df_generiques.iloc[:,0].to_list()
for i in generiques:
    # extract only the words in uppercase letter, which are the name of our interest
    regex = re.compile(r'[A-ZÀ-ÖØ-Þ]+')
    words = regex.findall(i)
    line = {"label": "DRUG", "pattern": [{"LOWER": j}for j in words]}
    json.dump(line, pattern)
    pattern.write("\n")


# add spécialités to dictionary
print(df_spécialités.shape)
df_generiques.drop_duplicates()
print(df_spécialités.shape)

spécialités = df_spécialités.iloc[:,0].to_list()
for i in spécialités:
    #separate elements with/

    # extract only the words in uppercase letter, which are the name of our interest
    regex = re.compile(r'[A-ZÀ-ÖØ-Þ]+')
    words = regex.findall(i)

    line = {"label": "DRUG", "pattern": [{"LOWER": j}for j in words]}
    json.dump(line, pattern)
    pattern.write("\n")

pattern.close()
# %%
