#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:47:11 2022

@author: christopher
"""


# DL-SpiQA Packages and functions

import os
import time
from datetime import datetime
import csv
import shutil
import numpy as np
import nrrd
import SimpleITK as sitk
import pandas as pd
from skimage.measure import label
from scipy import ndimage
import subprocess
from joblib import Parallel, delayed


def createFolders(root,foldNames):
    
    basePath = root
    for i in range(len(foldNames)):
        path = os.path.join(basePath,str(foldNames[i]))
        
        if not os.path.exists(path):
            os.mkdir(path)
            while not os.path.exists(path):
                time.sleep(0.1)
              
        basePath = os.path.join(basePath,path)

def moveOrCopy(srcFold,dstFold,mode='copy',ignore_errors=False):
    
    for f in os.listdir(srcFold):
    
        src = os.path.join(srcFold, f)
        dst = os.path.join(dstFold, f)
        
        try:
            if mode == 'copy':
                if os.path.isfile(src):
                    shutil.copy(src,dst)
                elif os.path.isdir(src):
                    shutil.copytree(src,dst)
                    
            if mode == 'move':
                shutil.move(src,dst)
        except:
            if not ignore_errors:
                print(f'Unable to {mode} {f}')

def cleanup(rootDir,foldList):
    for folder in foldList:
        if os.path.exists(os.path.join(rootDir,folder)):
            shutil.rmtree(os.path.join(rootDir,folder))
            

def bbox(vol):
    rows = np.any(vol, axis=(1,2))
    cols = np.any(vol, axis=(0,2))
    slices = np.any(vol, axis=(0,1))
    try:
        rmin, rmax = np.where(rows)[0][[0, -1]]
    except: 
        rmin = -1
        rmax = -1
    try:
        cmin, cmax = np.where(cols)[0][[0, -1]]
    except:
        cmin = -1
        cmax = -1
    try:
        smin, smax = np.where(slices)[0][[0, -1]]
    except:
        smin = -1
        smax = -1


    return [rmin, rmax, cmin, cmax, smin, smax]  

def getLargestCC(segmentation):
    labels = label(segmentation)
    try:
        assert( labels.max() != 0 )
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    except:
        largestCC = None
    return largestCC

def postprocessing(Y_pred,keepLargest=True):
    
    
    if len(Y_pred.shape)==3:
        Y = Y_pred
        Y = Y>0.5
        if np.sum(Y)>0:
            for k in range(Y.shape[2]):
                Y[:,:,k] = ndimage.binary_fill_holes(Y[:,:,k])
    
            Y = ndimage.binary_dilation(Y)
    
            if keepLargest:
                Y = getLargestCC(Y)
    
            Y = ndimage.binary_erosion(Y)
            
        Y_pred = Y

    return Y_pred

def niiTransform(vol, reverse=False):
    
    if reverse:
        return np.rot90(np.flipud(vol),axes=(2,0))
    else:
        return np.flipud(np.rot90(vol,axes=(0,2)))

def getBody(folder, largestCC=True, flip=True, reverse=False):
    
    fileList    = [x for x in os.listdir(folder) if x.endswith('.nii.gz')]
    bodyPath    = os.path.join(folder,'body.nii.gz')
    
    if os.path.exists(bodyPath):
        body    = sitk.ReadImage(bodyPath)
        dims    = body.GetSize()
        body    = sitk.GetArrayFromImage(body)
        if flip:
            body= niiTransform(body,reverse=reverse)
        if largestCC:
            body= getLargestCC(body)
            
    elif not os.path.exists(bodyPath) and len(fileList) > 0:
        print('Did not find body segmentation')
        im      = sitk.ReadImage(fileList[0])
        dims    = im.GetSize()
        body    = (np.ones(dims)).astype('uint8')
        if flip:
            body= niiTransform(body,reverse=reverse)
        if largestCC:
            body= getLargestCC(body)
            
    else:
        print(f'No TS outputs located in {folder}')
        return None, None
        
    return body, dims

def nifti2nrrd(file,outFolder,header=None,notBody=None):
    
    filename    = os.path.basename(file)
    
    niiData     = sitk.ReadImage(file)
    nrrdData    = sitk.GetArrayFromImage(niiData)
    if nrrdData.any():
        nrrdData    = np.flipud(np.rot90(nrrdData,axes=(0,2)))
        nrrdData    = (postprocessing(nrrdData,keepLargest=False)).astype('uint8')
        
        if not notBody is None:
            nrrdData[notBody]   = 0
            
        if not nrrdData.any():
            return
        
        nrrdName    = filename[:len(filename) - len('.nii.gz')]
        
        if header is None:
            nrrd.write(os.path.join(outFolder,'TS_'+nrrdName+'.nrrd'),nrrdData)
        else:
            trimHeader = ['type','encoding','endian']
            for item in trimHeader:
                if item in header.keys():
                    header.pop(item)
            nrrd.write(os.path.join(outFolder,'TS_'+nrrdName+'.nrrd'),nrrdData,header)
            
            
def nifti2nrrd_parallel(niftiFolder,outFolder,header=None,notBody=None):
    
    temp = []
    for niftiFile in sorted([x for x in os.listdir(niftiFolder) if '.nii.gz' in x]):
        niftiPath = os.path.join(niftiFolder,niftiFile)
        temp.append([niftiPath,outFolder,header,notBody])
       
    Parallel(n_jobs=-2)(delayed(nifti2nrrd)(t[0],t[1],t[2],t[3]) for t in temp)
    
def anyNumbers(s):
    for i in range(len(s)):
        if s[i].isnumeric():
            return True
    return False

def getVertebrae(s,spineList,sacrumList,sacrumNames):
    
    for sacName in sacrumList + sacrumNames:
        if sacName in s.lower():
            if not anyNumbers(s[3:(s.lower()).index(sacName)]):
                return ['sacrum']
    
    out = []
    idx = 0
    
    if 'BOS' in s.upper():
        out.append(spineList.index('C1'))
    
    for i in range(len(s)):
        
        if len(out) == 0:
            
            if s[i].isnumeric() and i > 2:
                if i + 1 < len(s) and s[i + 1].isnumeric():
                    l = 2
                else:
                    l = 1
                if s[i - 1:i + l] in spineList:
                    out.append(spineList.index(s[i - 1:i + l]))
                    idx = i
                    
                    for sacName in sacrumList + sacrumNames:
                        if sacName in s[i + l:].lower():
                            out.append(spineList.index('sacrum'))
                            return spineList[out[0]:out[1] + 1]
                    
                    if not anyNumbers(s[i + l:]):
                        return [spineList[out[0]]]
                    
                    continue
        
        if len(out) == 1:
            
            if s[i].isnumeric():
                
                if i == idx + 2:
                    if i + 1 < len(s) and s[i + 1].isnumeric():
                        l = 2
                    else:
                        l = 1
                    if s[i - 1] in ['-',':','_']:
                        out.append(spineList.index(s[idx - 1]+s[i:i + l]))
                        return spineList[out[0]:out[1] + 1]
            
                elif i == idx + 3:
                    if i + 1 < len(s) and s[i + 1].isnumeric():
                        l = 2
                    else:
                        l = 1
                    if s[i - 2:i].lower() == 'to':
                        out.append(spineList.index(s[idx - 1]+s[i:i + l]))
                        return spineList[out[0]:out[1] + 1]
                
                if i + 1 < len(s) and s[i + 1].isnumeric():
                    l = 2
                else:
                    l = 1
                if s[i - 1:i + l] in spineList:
                    out.append(spineList.index(s[i - 1:i + l]))
                    return spineList[out[0]:out[1] + 1]
                elif s[i - 1:i + l] in sacrumList:
                    out.append(spineList.index('sacrum'))
                    return spineList[out[0]:out[1] + 1]

def checkVolume(tVert, nVerts, p = 0.5, k = 1, kSup = 1, kInf = 1):
    
    if np.abs(bbox(tVert)[5] - tVert.shape[2]) <= 2 or bbox(tVert)[4] <= 2:
        return 0
    
    if len(nVerts) == 1:
            
        if k*np.count_nonzero(tVert) > (1 + p)*np.count_nonzero(nVerts[0]):
            return 1
        elif k*np.count_nonzero(tVert) < (1 - p)*np.count_nonzero(nVerts[0]):
            return -1
        else:
            return 0
        
    if len(nVerts) == 2:
        
        if np.abs(bbox(nVerts[0])[5] - tVert.shape[2]) <= 2:
            
            if k*np.count_nonzero(tVert) > kInf*(1 + p)*np.count_nonzero(nVerts[1]):
                return 1
            elif k*np.count_nonzero(tVert) < kInf*(1 - p)*np.count_nonzero(nVerts[1]):
                return -1
            else:
                return 0
            
        elif bbox(nVerts[1])[4] <= 2:
            
            if k*np.count_nonzero(tVert) > kSup*(1 + p)*np.count_nonzero(nVerts[0]):
                return 1
            elif k*np.count_nonzero(tVert) < kSup*(1 - p)*np.count_nonzero(nVerts[0]):
                return -1
            else:
                return 0
        
        avgVol = (kSup*np.count_nonzero(nVerts[0]) + kInf*np.count_nonzero(nVerts[1]))/2
        if k*np.count_nonzero(tVert) > (1 + p)*avgVol:
            return 1
        if k*np.count_nonzero(tVert) < (1 - p)*avgVol:
            return -1
        else:
            return 0

def spineVolumesAll(nrrdPath, vertsLog, tsVertList, cRegion=True, p=0.5):
    
    vertList = []
    for i, v in enumerate(tsVertList):
        
        if 'TS_' + v + '.nrrd' in os.listdir(nrrdPath):
            vertList.append(v)
    
    volList = {}
    volSum  = 0
    
    for i, v in enumerate(vertList):
        
        key = v
        if 'vertebrae' in key:
            key = v.split('_')[1]
        
        volPath = os.path.join(nrrdPath,'TS_' + v + '.nrrd')
        if os.path.exists(volPath):
            vol     = nrrd.read(volPath)[0]
            
            if not (np.abs(bbox(vol)[5] - vol.shape[2]) <= 2 or bbox(vol)[4] <= 2):
                volList[key] = np.count_nonzero(vol)
                volSum += volList[key]
                
    vMed    = vertsLog.loc[vertsLog['Label'] == 'Median']
    vStd    = vertsLog.loc[vertsLog['Label'] == 'Standard Deviation']
    
    vNorm   = {}
    vSum    = 0
    for key, value in volList.items():
        vNorm[key] = np.array([vMed[key].iloc[0], vStd[key].iloc[0]])
        vSum += vMed[key].iloc[0]
        
        value /= volSum
        volList[key] = value
        
    volFlags = {}
    for key, value in vNorm.items():
        vNorm[key] /= vSum
        
        if volList[key] < vNorm[key][0] - vNorm[key][1]:
            volFlags[key] = -1
        elif volList[key] > vNorm[key][0] + vNorm[key][1]:
            volFlags[key] = 1
        else:
            volFlags[key] = 0
    
    if cRegion and len([i[0] for i in volList.items() if i[0][0] == 'C']) > 0:
        for i, c in enumerate([i[0] for i in volList.items() if i[0][0] == 'C']):
            k = 1
            kSup = 1
            kInf = 1
            
            if c == 'C1':
                k = 1.236
                volCmpr = kInf*volList['C2']
                    
            elif c == 'C2':
                k = 0.7720
                if 'C1' in volList.keys():
                    volCmpr = (kSup*volList['C1'] + kInf*volList['C3'])/2
                else:
                    volCmpr = kInf*volList['C3']
                
            elif c == 'C3':
                kSup = 0.7720
                if 'C2' in volList.keys():
                    volCmpr = (kSup*volList['C2'] + kInf*volList['C4'])/2
                else:
                    volCmpr = kInf*volList['C4']
                
            elif c == 'C7':
                if 'C6' in volList.keys() and 'T1' in volList.keys():
                    volCmpr = (kSup*volList['C6'] + kInf*volList['T1'])/2
                elif not 'C6' in volList.keys():
                    volCmpr = kInf*volList['T1']
                elif not 'T1' in volList.keys():
                    volCmpr = kSup*volList['C6']
                
            else:
                cSup = 'C'+str(int(c[1:]) - 1)
                cInf = 'C'+str(int(c[1:]) + 1)
                
                if cSup in volList.keys() and cInf in volList.keys():
                    volCmpr = (kSup*volList[cSup] + kInf*volList[cInf])/2
                elif not cSup in volList.keys():
                    volCmpr = kInf*volList[cInf]
                elif not cInf in volList.keys():
                    volCmpr = kSup*volList[cSup]
                
            if k*volList[c] > (1 + p)*volCmpr:
                volFlags[c] = 1
            elif k*volList[c] < (1 - p)*volCmpr:
                volFlags[c] = -1
            else:
                volFlags[c] = 0
    
    return volFlags


def getDice(x,y,eps=1e-4):
    union = np.sum(x*y)
    return 2*union/(np.sum(x)+np.sum(y)+eps)


def getDVH(X,dDose = 50., noBins=None,maxDose = None):
    if maxDose is None:
        maxDose = np.max(X)
    if len(X)==0:
        return None,None
    else:
        if noBins is not None:
            d = np.linspace(0,maxDose,noBins)
        else:
            d = np.arange(0,(np.ceil(maxDose/dDose)+1)*dDose,dDose)
        h = np.zeros_like(d)
        n = len(X)
        for i,dd in enumerate(d):
            h[i] = np.sum(X>=dd)/n
        return d,h
    
def compareStructure(pid,struct,nrrdFolder): 
    
    try:
        filename = os.path.join(nrrdFolder,pid,'RS','AI_'+struct+'.nrrd')
        a  = nrrd.read(filename)[0]
        
        filename = os.path.join(nrrdFolder,pid,'RS',struct+'.nrrd')
        b  = nrrd.read(filename)[0]
        
        dice = getDice(a,b)
        return dice
    
        print(f'{pid} {struct}: dice={dice:.2}')
    except:
        return None

def getDoseInfo(pid,struct,nrrdFolder,dose,dDose=50):
    
    try:
        
        
        if type(struct)==list:
            a = None
            for st in struct:
                filename = os.path.join(nrrdFolder,pid,'RS',st+'.nrrd')
                data  = nrrd.read(filename)
                if a is None:
                    a = data[0]
                    header = data[1]
                else:
                    a+=data[0]
                
                a[a>1] = 1
            
        else:
            if struct.endswith('.nrrd'):
                filename = os.path.join(nrrdFolder,pid,'RS',struct)
            else:
                filename = os.path.join(nrrdFolder,pid,'RS',struct+'.nrrd')
            data  = nrrd.read(filename)
            a = data[0]
            header = data[1]
        dV = header['space directions'][0,0]*header['space directions'][1,1]*header['space directions'][2,2]
        vol = np.sum(a*dV)/1000
            

        d  = dose
        maxDoseDVH = np.ceil(np.max(d))
        dose = d[a==1]
            
        minDose = np.min(dose)
        maxDose = np.max(dose)
        meanDose = np.mean(dose)
            
        dvh = getDVH(dose,dDose,maxDose=maxDoseDVH)
        
        return {'Volume':vol, 'min':minDose,'max':maxDose,'mean':meanDose,'dvh':dvh}

    except:
        print('Export Error', end = '')
        return None

def exportDVH(dvhFolder,nrrdFolder,doseFolder=None,pid=None,addDate=False,booleanStruct = {'Coronary All':['AI_LAD','AI_PDA','AI_RCA','AI_LCX','AI_LM']},dDose=50):
    
    if not os.path.exists(dvhFolder):
        os.mkdir(dvhFolder)
    if pid is None:
        pids = [ file for file in os.listdir(nrrdFolder)]
    else:
        pids = [pid] 
        
    print('DVHs: ', end = ' ')
    for pid in pids:    
        data = []
        
        if os.path.exists(os.path.join(nrrdFolder,pid,'DOSE','DOSE.nrrd')):
            
            if doseFolder is None:
                dose = nrrd.read(os.path.join(nrrdFolder,pid,'DOSE','DOSE.nrrd'))[0]
            else:
                dose = nrrd.read(doseFolder)[0]
            dose=100.*dose
            print(f'-> max Dose {np.max(dose):.2f} cGy',end = ' ')
            for file in os.listdir(os.path.join(nrrdFolder,pid,'RS')):
                name = file.replace('.nrrd','')
                print(f'{name}',end=' ')
                dvh=getDoseInfo(pid,file,nrrdFolder,dose,dDose)
                dice = compareStructure(pid,name,nrrdFolder)
         
                
                try:
                    temp = []
                    temp.append(name)
                    temp.append("{:.2f}".format(dvh['Volume']))
                    if dice is not None:
                        temp.append("{:.2f}".format(dice))
                    else:
                        temp.append("N/A")
                    temp.append("{:.2f}".format(dvh['min']))
                    temp.append("{:.2f}".format(dvh['max']))
                    temp.append("{:.2f}".format(dvh['mean']))
                    for i in range(len(dvh['dvh'][0])):
                        temp.append("{:1.2f}".format(dvh['dvh'][1][i]))
                    data.append(temp)
                except:
                    continue
            
            if booleanStruct is not None:
    
                for bs,cs in booleanStruct.items():
                    dvh=getDoseInfo(pid,cs,nrrdFolder,dose,dDose)
                    
                    temp = []
                    temp.append(bs)
                    temp.append("{:.2f}".format(dvh['Volume']))
                    temp.append("N/A")
                    temp.append("{:.2f}".format(dvh['min']))
                    temp.append("{:.2f}".format(dvh['max']))
                    temp.append("{:.2f}".format(dvh['mean']))
        
                    for i in range(len(dvh['dvh'][0])):
                        temp.append("{:1.2f}".format(dvh['dvh'][1][i]))
                    data.append(temp)
    
                
            print('Done')   
            try:
                header=['StructureID','Volume [cc]','Dice','min Dose [cGy]','max Dose [cGy]','mean Dose [cGy]']
                for i in range(len(dvh['dvh'][0])):
                        header.append("{:.2f}".format(dvh['dvh'][0][i])) 
            
                if (addDate):
                    now = datetime.now()
                    date_time = now.strftime("%Y_%m_%d_")
                    csvFile = os.path.join(dvhFolder,date_time+pid+'_dvh.csv')
                else:
                    csvFile = os.path.join(dvhFolder,pid+'_dvh.csv')
                with open(csvFile, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    
                    writer.writerow(header)
                    for row in data:
                        writer.writerow(row)
            except:
                print('Unable to write DVH')
        else:
            print('Unable to find Dose file')

def SpineQA(dataPullPath, dataPushPath, workingFolders, spineDict, gpu=0):
    
    # Initialize working environment
    print('Initializing DL-SpiQA')
    
    workingDir  = workingFolders['root']
    if not os.path.exists(workingDir):
        os.mkdir(workingDir)
        while not os.path.exists(workingDir):
            time.sleep(0.1)
    
    dataPullLoc     = os.path.join(workingDir,'data_pull')
    inputDataPath   = os.path.join(workingDir,'input_data')
    outputDataPath  = os.path.join(workingDir,'output_data')
    tsFolder        = os.path.join(workingDir,'TS_Output')
        
    # Initialize variables
    
    spineList   = spineDict['spine_list']
    sacrumList  = spineDict['sacrum_list']
    sacrumNames = spineDict['sacrum_names']
    tsVertList  = spineDict['TS_list']
    aiModelPath = spineDict['ai_model']
    
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    
    # Download and process patient data sequentially as a batch process
    
    patList     = sorted([x for x in os.listdir(dataPullPath) if os.path.isdir(os.path.join(dataPullPath,x))])
    for pCount, pat in enumerate(patList):
        
        print(f'Processing {pat}')
        for key, folder in workingFolders.items():
            if key == 'root':
                continue
            createFolders(workingDir,[folder])
        
        # Download patient NRRD data
        print('Downloading NRRD data')
        patFolder   = os.path.join(dataPullLoc,pat)
        shutil.copytree(os.path.join(dataPullPath,pat),patFolder)
        
        # Convert CT volume to NIfTI format for TotalSegmentator
        ctFile      = os.path.join(patFolder,'CT','CT.nrrd')
        ctNifti     = sitk.ReadImage(ctFile)
        ctNiiFile   = os.path.join(inputDataPath,'ct.nii.gz')
        sitk.WriteImage(ctNifti, ctNiiFile)
        
        # Run the TotalSegmentator model to acquire vertebral segmentations
        print('Generating vertebral segmentations')
        cmdList = [aiModelPath,'-i',ctNiiFile,'-o',tsFolder,'--roi_subset'] + tsVertList
        subprocess.run(cmdList)
        
        # Acquire body segmentation using TotalSegmentator
        print('Generating body segmentation')
        cmdList = [aiModelPath,'-i',ctNiiFile,'-o',inputDataPath,'-ta','body']
        subprocess.run(cmdList)
        
        # Convert TS output NIfTI files to NRRD
        print('Converting TotalSegmentator outputs to NRRD')
        patOut  = os.path.join(outputDataPath,pat)
        rsFolder= os.path.join(patOut,'RS')
        createFolders(outputDataPath,[pat,'RS'])
        
        header  = nrrd.read_header(ctFile)
        body    = getBody(inputDataPath)
        notBody = (body == 0)
        nifti2nrrd_parallel(tsFolder, rsFolder, header=header, notBody=notBody)
        
        # Copy vertebral segmentations, dose, and image data to output folder
        moveOrCopy(patFolder,patOut)
        
        # Calculate DVHs
        print('Exporting DVHs')
        exportDVH(patOut,outputDataPath,booleanStruct=None)
        dvhFile = os.path.join(patOut,pat+'_dvh.csv')
        
        # Get RT plan data
        rtDataFile  = os.path.join(patOut,'RT Plan Data.txt')
        file    = open(rtDataFile,'r')
        lines   = file.readlines()
        file.close()
        
        rtPlanLabel = (lines[0].split(': '))[1].split('\n')[0]
        vertList    = getVertebrae(rtPlanLabel,spineList,sacrumList,sacrumNames)
        prDose      = float((lines[1].split(': '))[1].split('\n')[0])
        
        # Check for dose or volume discrepancies
        flags   = {}
        flags['Target Vertebrae']   = []
        flags['Underdose Check']    = []
        flags['Overdose Check']     = []
        flags['Volume Check']       = []
        
        df_dvh  = pd.read_csv(dvhFile)
        vertsLog= pd.read_csv(os.path.join(dataPullPath,'vert_volume_statistics.csv'))
        
        flags['Target Vertebrae']   = vertList
        volDict = spineVolumesAll(rsFolder,vertsLog,tsVertList)
        for vKey in volDict.keys():
            if not volDict[vKey] == 0:
                flags['Volume Check'].append(vKey)
                
        rejectCols = ['StructureID','Volume [cc]','Dice','min Dose [cGy]','max Dose [cGy]','mean Dose [cGy]']
        vDict = {'PatientID':patFolder, 'PlanLabel':rtPlanLabel, 'PrescribedDose':prDose}
        
        colList = df_dvh.columns.tolist()
        doseList = []
        for d in colList:
            if not d in rejectCols:
                doseList.append(float(d))
        doseList = np.array(sorted(doseList))
        
        vCount = 0
        for vert in spineList:
            if vert in [x.split('_')[-1] for x in df_dvh['StructureID'].tolist()]:
                if vert == 'sacrum':
                    v = 'TS_' + vert
                else:
                    v = 'TS_vertebrae_' + vert
            else:
                continue
            
            
            vCount += 1
            vert_df = df_dvh.loc[df_dvh['StructureID'] == v]
            d50p = 100 * prDose / 2
            
            dIdx = np.argmin(np.abs(doseList - d50p))
            dIdx += len(rejectCols)
            
            dCol = colList[dIdx]
            v50p = vert_df.iloc[0][dCol]
            
            vvol = vert_df.iloc[0]['Volume [cc]']
            
            vDict['Structure_'+str(vCount)] = vert
            vDict['V50_'+str(vCount)] = v50p
            vDict['Volume[cc]_'+str(vCount)] = vvol
            
            if vert in vertList and v50p < 0.5:
                flags['Underdose Check'].append(vert)
            elif (not vert in vertList) and (v50p > 0.5):
                flags['Overdose Check'].append(vert)
                
        # Generate DL-SpiQA report
        print('Generating DL-SpiQA report')
        qa_status = np.logical_and.reduce((flags['Underdose Check']==[],flags['Overdose Check']==[],flags['Volume Check']==[]))
        
        rprtPath = os.path.join(dataPushPath,'DL-SpiQA Report.txt')
        f = open(rprtPath,'a')
        wLines  = []
        
        wLines.append('SPINE QA SUMMARY FOR ' + pat + '\n')
        
        if qa_status:
            wLines.append('QA Status: PASS\n')
        else:
            wLines.append('QA Status: FLAGS RAISED\n')
        
        wLines.append('RT Plan Label: ' + rtPlanLabel + '\n')
        vertStr = ''
        for v in vertList:
            vertStr += v + ' '
        wLines.append('Target Vertebrae: ' + vertStr + '\n')
            
        v1 = ''
        for v in flags['Underdose Check']:
            v1 += v + ' '
        if not v1 == '':
            wLines.append('Underdosed vertebrae: '+v1+'\n')
        
        v2 = ''
        for v in flags['Overdose Check']:
            v2 += v + ' '
        if not v2 == '':
            wLines.append('Overdosed vertebrae: '+v2+'\n')
        
        v3 = ''
        for v in flags['Volume Check']:
            v3 += v + ' '
        if not v3 == '':
            wLines.append('Volume discrepancies: '+v3+'\n')
        
        wLines.append('\n')
        
        # Print a summary of DL-SpiQA results
        print('\n\n')
        for line in wLines:
            print(line)
            f.write(line)
        
        f.close()
        
        # Delete leftover files in working directory
        print('Cleaning up local files')
        cleanup(workingDir,[workingFolders['pull'],workingFolders['input'],workingFolders['output'],workingFolders['TotSeg']])

#%%

### NOTE: Please edit the following paths.
### dataPullPath points to a data input folder containing deidentified patient data in NRRD format.
### dataPushPath points to a destination folder to store results (by default, this is the same as the input folder).
### workingRoot specifies an empty folder on the local machine to serve as a working directory for file handling.
### aiModelPath points to the location of the TotalSegmentator model.

dataPullPath    = '/home/christopher/Documents/misc/spine_data'
dataPushPath    = dataPullPath
workingRoot     = '/home/christopher/Documents/Total_Segmentator'
aiModelPath     = '/home/christopher/environments/totseg2/totseg2/bin/TotalSegmentator'


# Folders for the working directory
workingFolders  = {'root'   :'/home/christopher/Documents/Total_Segmentator/',
                   'pull'   :'data_pull',
                   'input'  :'input_data',
                   'output' :'output_data',
                   'TotSeg' :'TS_Output'}


# Useful lists for spine enumeration
spineList   = ['C' + str(c+1) for c in range(7)] + ['T' + str(t+1) for t in range(12)] + ['L' + str(l+1) for l in range(5)] + ['sacrum']
sacrumList  = ['S' + str(s+1) for s in range(5)]
sacrumNames = ['sacrum','sac','sacr','sacru','sarcu','coccyx','ccyx']
tsVertList  = ['vertebrae_'+x for x in spineList[:-1]] + ['sacrum']

spineDict   = {'spine_list'     : spineList,
               'sacrum_list'    : sacrumList,
               'sacrum_names'   : sacrumNames,
               'TS_list'        : tsVertList,
               'ai_model'       : aiModelPath}

start_time = time.time()
SpineQA(dataPullPath, dataPushPath, workingFolders, spineDict, gpu=0)
print("Batch processing completed in %.2f seconds" % (time.time() - start_time))