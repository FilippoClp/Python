# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:43:06 2018

@author: nicoleg
"""
# INPUT 
# INPUT: Put the name of the head of the group
# INPUT 

# "NL0001"



#IT0001967291493
#IT0000102998324
#AT43000
#DEFDI0954
#SK31320155
#AT19940
#DE01127

nomi=["DE00001"];  
#"IT0000102484824",
# "ES0049"      
# "DE00003"


'FR439208190',
"IT0000807584025",
"IT0000203426147",
"IT0000375525937",
"IT0000101244932",
"IT0000102341108",
"IT0000103225010",
"IT0002819303966",
"IT0000101247669",
"IT0000126876530",
"IT0000104747575",
"IT0000101262255",
"DE00001","DE03472",
"LT40100",
"LT70400",
"ESNCG",
"SI5026024",
"LV000410",
"NLABN",
"FR45129",
"IEAIBPLC",
"GR014",
"EE10004252",
"EE10060701",
"BE0404476835",
"ES0182",
"PT10",
"PT33",
"ES0240",
"ES0081",
"ES0487",
"ES0075",
,
"CY110002",
"MTCIVALL",
"ES0128",
"BE0403212172",
"LUB00001",
"FR30588",
"DE00317",
"BE0403201185",
"ES7865",
"FR30004",
"FR12589",
"FR11380",
"PT35"
"FR22040",
"NL600",
"CY110003",
"FI17307447",
"DE00313",
"DE01135",
"DE03249",
"BE0458548296",
"DE01121",
"AT20100",
"DEBA2859945",
"GR026",
"ES2100",
"FR16188",
"FR30006",
"DEBA1137415",
"CY110005",
"MTCIHSBC",
"FR30056",
"DE05749",
"ES2085"]

# some toolboxes
import requests
import xml.etree.ElementTree as ET
import math
import pandas as pd
import csv

####################################3
#### CALL RIAD on the group
#####################################33

#lab_prj_sds.csdb_query

for index in range(0,len(nomi)-1): 
    namegroup_riad = nomi[index]
    header = "https://exdwp81.ecb01.ecb.de:8080/ws/RIAD-WSREST/wsrest/"
    source = "groups/sdd/current?"
    retrieve = "rp=grp_hd_nm_entty,grp_snpsht_dt,grp_typ&rpe=chld_entty_riad_id,chld_lvl,chld_pstn,prnt_entty_riad_id,typ_rltnshp&"
    filtro = "f.grp_hd_id.in=" 
    queries = header+source+retrieve+filtro+namegroup_riad          

    res = requests.get(queries, auth=('EUNICOLEG', ''), verify=False, stream=True)        
    ns = '{http://www.ecb.int/schema/RIAD/sdd/groups}'  # qui definisco il namespace
    with open(namegroup_riad+'.xml', 'wb') as file:
        file.write(res.content)
    tree = ET.parse(namegroup_riad+'.xml')
    root1 = tree.getroot()
    if res.content == b'providerResponse=' : 
       pippo = namegroup_riad
    elif not (root1.findall(ns + 'results/' + ns +'group')):
       pippo = namegroup_riad
    else :
#    doing parsing on groups.xml         
       item = root1.findall(ns + 'results/' + ns +'group')
#select only group: was group B
       groupstru = item[0]
       groupstru1=groupstru.find(ns+'groupElements')
       pippo = []
       count = 0
       enda = len(groupstru1)
       for sub_node in range(0,enda):
           subba = groupstru1[sub_node]
           tagga = subba.find(ns + 'chld_entty_riad_id')
           count +=1 
           pippo.append(tagga.text)     

# Pippo now contains the list of 
       header = "https://exdwp81.ecb01.ecb.de:8080/ws/RIAD-WSREST/wsrest/"
       source = "orgunits/sdd/current?"
       retrieve = "rp=entty_riad_id,entty_riad_cd,nm_entty,cty&"
       filtro = "f.entty_riad_id.in=" 
       dictionary = {}
       nfiles = math.ceil(len(pippo)/100) 
       astart = 0
       aend = 99
       for index in range(0,nfiles-1):
           entity = pippo[astart:aend]
        # adjust here and launch
           if index < nfiles-2:
               astart += 100
               aend += 100
               print(astart)
           else: 
               windo = len(pippo) - astart
               astart += windo 
               aend += windo 
           queries = header+source+retrieve+filtro+','.join(entity)          
           res = requests.get(queries, auth=('EUNICOLEG', 'pwd'), verify=False, stream=True)        
           with open('manybanks.xml', 'wb') as file:
               file.write(res.content)
   #    tree = ET.parse('manybanks.xml')
#parsing using the dictionaries ?
           tree = ET.parse('manybanks.xml')
           root = tree.getroot()
           ns = '{http://www.ecb.int/schema/RIAD/sdd/orgUnits}'  # qui definisco il namespace
           results = root.find(ns + 'results') # vado al nodo results
# per ogni orgUnit stampo il valore dell'attributo entty_riad_cd
           count =1
           idcode = []
           cdcode = []
           for item in results.findall(ns + 'orgUnit'):
               for child in item:
                   if child.tag == ns + 'entty_riad_id': # controllo che l'attributo sia quello giusto
                       idcode.append(child.text)  
                   elif child.tag == ns + 'entty_riad_cd':
                       cdcode.append(child.text)  
                       count +=1
               dictionary1 = dict(zip(idcode, cdcode))
           dictionary.update(dictionary1)
# save in a .csv file 
           a=dictionary.values()
           with open(namegroup_riad+'simple.csv', 'w', newline = '') as file:
               wr = csv.writer(file)
               wr.writerow(a)






#now use it in CSDB
# 
import pyodbc, os, sys
cnxn = pyodbc.connect('DSN=DISC DP Impala 64bit',autocommit=True)
cursor = cnxn.cursor()

pippo = """'2015-12-31 00:00:00'"""
# to be checked maybe not needed
otherdate = """'2016-01-31 00:00:00'"""
debttype = """'D.%'"""
debttype18 = """'D.18%'"""
cento = """'100'"""
uno = """'1'"""
pip = 'DE00003'
mfi_list = pip


LEND = """'LEND'"""
CANC = """'CANC'"""
listamfi = dictionary.values()
listamfi = 'DE00001'


# '
import pandas as pd
sql = 'select residualmaturity,amountoutstanding_eur,issuerexternalcode_mfi,externalcode_isin as isin,idircountry as country,idircurrency_nominal, isactive, va_securitystatus,va_isinsec,idmaturitydate,idirorganisationaliastype_is,seniority' 
sql = sql + ' from crp_csdb.csdb_inst_fact_ncbex_revsd_optimize_tmp' 
sql = sql + ' where (idirdebttype LIKE ' + debttype + ')'  
sql = sql + ' and (idirdebttype not LIKE ' + debttype18 + ')'  
sql = sql + ' and (va_securitystatus =' + cento + ')'  
sql = sql + ' and (va_isinsec =' + uno + ')'  
sql = sql + ' and correction_date =' + pippo  
sql = sql + ' and idloaddate_dim =' + otherdate
sql = sql + ' and amountoutstanding_eur is not null ' 
sql = sql + ' and issuerexternalcode_mfi = ' + "'" + str(mfi_list) + "'"


#sql = sql + 'order by report_agent_lei,cntp_lei,cntp_loc_code, cntp_sector,trade_date, coll_isin;' 
df = pd.read_sql(sql, cnxn)
df.columns #check column names






#/*
#import xml.etree.ElementTree as ET
#tree = ET.parse('orgunit.xml')
#root = tree.getroot()
#
#ns = '{http://www.ecb.int/schema/RIAD/sdd/orgUnits}'  # qui definisco il namespace
#
#results = root.find(ns + 'results') # vado al nodo results
#
## per ogni orgUnit stampo il valore dell'attributo entty_riad_cd
#for sub_node in results.find(ns + 'orgUnit'):
#    if sub_node.tag == ns + 'entty_riad_cd': # controllo che l'attributo sia quello giusto
#        print(sub_node.text)
#
#      
##    doing parsing on orgunit.xml         
#import xml.etree.ElementTree as ET
#tree = ET.parse('orgunit.xml')
#root = tree.getroot()
#
#ns = '{http://www.ecb.int/schema/RIAD/sdd/orgUnits}'  # qui definisco il namespace
#
#results = root.find(ns + 'results') # vado al nodo results
#
## per ogni orgUnit stampo il valore dell'attributo entty_riad_cd
#
#count =1
#pippo = []
#for sub_node in results.find(ns + 'orgUnit'):
#    if sub_node.tag == ns + 'entty_riad_cd': # controllo che l'attributo sia quello giusto
##        pippo[count] = sub_node.text
#        count += 1
#        pippo.append(sub_node.text)  
#
## parsing on orguint.xml
