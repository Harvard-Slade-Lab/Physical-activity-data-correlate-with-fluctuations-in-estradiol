#!/usr/bin/python

import numpy as np
import pandas as pd
from scipy import stats



def get_cycleorder(subject, model_outputs, metadata, phasedata, ref='Menses'):
    # get participant's dates and phase data
    part_data = metadata.loc[metadata['Subject'] == subject]
    part_phases = phasedata.loc[phasedata['usub'] == subject]

    # get dates of menses start, menses end, and ovulation
    study_start = pd.to_datetime(part_phases['startDate'].values[0])
    menses_onset = pd.to_datetime(part_data['MenstruationOnset1'].values[0])    
    menses_offset = pd.to_datetime(part_data['MenstruationOffset1'].values[0])
    ovulation = pd.to_datetime(part_data['Ovulation1'].values[0])

    # get indices of the above dates
    timestamps = model_outputs[subject]['gait_timestamps']
    try:
        mensesonset_key = int([key for key in timestamps if timestamps[key][0].date() == menses_onset.date()][0])
    except IndexError:
        mensesonset_key = min(model_outputs[subject]['TEE_J'].keys())

    mensesoffset_key = int([key for key in timestamps if timestamps[key][0].date() == menses_offset.date()][0])-1
    if isinstance(ovulation, pd._libs.tslibs.nattype.NaTType) == True:
        print('no ovulation reported')
        ovulation_key = None
    else:
        ovulation_key = int([key for key in timestamps if timestamps[key][0].date() == ovulation.date()][0])
    
    original_order = list(model_outputs[subject]['TEE_J'].keys())
    
    # 1) REORDER STUDY DAYS TO MATCH CYCYLE PROGRESSION FROM MENSES TO LUTEAL
    if ref == 'Menses':
        reordered_idx = []
        if min(model_outputs[subject]['TEE_J'].keys()) == 0:
            for idx in original_order[mensesonset_key:]:
                reordered_idx.append(idx)
            for key in timestamps:
                if key not in reordered_idx:
                    reordered_idx = np.append(reordered_idx, key)
        elif min(model_outputs[subject]['TEE_J'].keys()) == 1:
            for idx in original_order[mensesonset_key-1:]:
                reordered_idx.append(idx)
            for key in timestamps:
                if key not in reordered_idx:
                    reordered_idx = np.append(reordered_idx, key)
        return reordered_idx, mensesonset_key, mensesoffset_key
    
    # 2) REORDER STUDY DAYS TO MATCH CYCYLE PROGRESSION STARTING WITH OVULATION
    elif ref == 'Ovulation':
        if ovulation_key == None:
            reordered_idx = []
        else:        
            if mensesonset_key < ovulation_key:
                menses_onset_from_LH = mensesonset_key - ovulation_key
                
                N = len(model_outputs[subject]['gait_timestamps'])

                # values to the left
                menses_to_ovulation = np.arange(menses_onset_from_LH, 0)
                
                #values to the right
                ovulation_to_laststudyday = np.arange(0, N-ovulation_key+1)
                
                # pre menses
                wrapdays = np.arange(ovulation_to_laststudyday[-1]+1, ovulation_to_laststudyday[-1]+mensesonset_key)

                reordered_idx = np.concatenate((wrapdays, menses_to_ovulation, ovulation_to_laststudyday)).astype(int)
            else:
                # values left of ovulation
                start_to_ovulation = np.arange(0-ovulation_key+1, 0)

                # values right of ovulation
                ovulation_to_menses = np.arange(0, original_order.index(mensesonset_key)-original_order.index(ovulation_key))

                # wrap back to menses onset 
                try:
                    menses_val = original_order.index(mensesonset_key) - len(original_order) + start_to_ovulation[0]
                    menses_to_start = np.arange(menses_val, start_to_ovulation[0])
                except IndexError:
                    menses_to_start = np.arange(original_order.index(mensesonset_key) - len(original_order), 0)

                reordered_idx = np.concatenate((start_to_ovulation, ovulation_to_menses, menses_to_start)).astype(int)
                
        return reordered_idx, ovulation_key
    
def dct_to_df(model_estimates, subjects, metadata):
    """Get number of steps and AEE per day, normalized by body weight, and store in a dataframe in order of study progression."""
    AEE_sum_kJkg = {}
    N_steps = {}
    for subject in subjects: 
        # get participant's weight
        part_data = metadata.loc[metadata['Subject'] == subject]
        weight_kg = part_data['Weight_kg'].values[0]
        
        # sum AEE (kJ/kg) and number of steps per day
        AEE_sum_kJkg[subject] = []
        N_steps[subject] = []
        recorded_days = model_estimates[subject]['AEE_J'].keys()
        for day in recorded_days:
            AEE_per_day = np.sum(model_estimates[subject]['AEE_J'][day])/1000
            AEE_sum_kJkg[subject].append(AEE_per_day/weight_kg)
            N_steps[subject].append(len(model_estimates[subject]['AEE_J'][day]))

    # store data in dataframes
    AEE_kJkg_studyorder = pd.DataFrame.from_dict(AEE_sum_kJkg, orient='index')
    N_steps_df = pd.DataFrame.from_dict(N_steps, orient='index')
    # rename columns
    AEE_kJkg_studyorder.columns = np.arange(1, len(AEE_kJkg_studyorder.columns)+1)
    N_steps_df.columns = np.arange(1, len(N_steps_df.columns)+1)
    return AEE_kJkg_studyorder, N_steps_df

def sort_df(AEE_kJkg_studyorder, N_steps_df, model_outputs, metadata, phasedata):
    """Sort columns in df such  luteal phase."""
    # LABEL PHASES BASED ON METADATA.CSV DATES
    AEE_sum_kJkg_cycleorder = {}
    N_steps_cycleorder = {}
    sorted_model_output = {}
    cycle_order_days = {}

    for subject in AEE_kJkg_studyorder.index:
        # initiate vars
        cycle_order_days[subject] = {}
        sorted_model_output[subject] = {}

        # determine order of days w.r.t. ovulation timing
        cycle_order_LH, ovulation_key = get_cycleorder(subject, model_outputs, metadata, phasedata, ref='Ovulation')
        
        # sort in ascending order so that list starts with menses onset 
        cycle_order_LH = sorted(cycle_order_LH)
            
        # determine order of days w.r.t. menses timing
        cycle_order_menses, mensesonset_key, mensesoffset_key = get_cycleorder(subject, model_outputs, metadata, phasedata, ref='Menses')

        # reorder dictionary (model_outputs) based on menses onset
        for metric in model_outputs[subject].keys():
            if metric != 'drop_studydays':
                sorted_model_output[subject][metric] = {day: model_outputs[subject][metric][day] for day in cycle_order_menses}
        # estimate timing of luteal phase if ovulation is not reported
        luteal_key = None
        if ovulation_key == None:
            luteal_idx = int(len(model_outputs[subject]['gait_timestamps'].keys())/2)
            luteal_key = cycle_order_menses[luteal_idx]
            
        # add phase labels to sorted dictionary
        phase_label = []
        prev = None
        for i in cycle_order_menses:
            if i == mensesonset_key:
                phase_label.append('Early Follicular')
                prev = 'Early Follicular'
                continue
            elif i == mensesoffset_key:
                phase_label.append('Late Follicular')
                prev = 'Late Follicular'
                continue
            elif i == ovulation_key:
                phase_label.append('Ovulation')
                prev = 'Ovulation'
                continue
            elif i == luteal_key:
                phase_label.append('Luteal')
                prev = 'Luteal'
                continue

            if prev == 'Early Follicular':
                phase_label.append('Early Follicular')
                continue
            if prev == 'Late Follicular':
                phase_label.append('Late Follicular')
                continue 
            if prev == 'Ovulation':
                phase_label.append('Luteal')
                prev = 'Luteal'
                continue
            if prev == 'Luteal':
                phase_label.append('Luteal')
                continue
            
        # store phase labels in same dictionary as energy estimates
        sorted_model_output[subject]['phase'] = phase_label
        sorted_model_output[subject]['study_day'] = cycle_order_menses
        sorted_model_output[subject]['time_from_LH'] = cycle_order_LH
        
        # reorder AEE and steps dfs to match cycle order
        AEE_sum_kJkg_cycleorder[subject] = [AEE_kJkg_studyorder[day][AEE_kJkg_studyorder.index == subject].values[0] for day in cycle_order_menses]
        N_steps_cycleorder[subject] = [N_steps_df[day][N_steps_df.index == subject].values[0] for day in cycle_order_menses]
    
    AEE_kJkg_cycleorder = pd.DataFrame.from_dict(AEE_sum_kJkg_cycleorder, orient='index')
    AEE_kJkg_cycleorder.columns = np.arange(1, len(AEE_kJkg_cycleorder.columns)+1)
    N_steps_cycleorder = pd.DataFrame.from_dict(N_steps_cycleorder, orient='index')
    N_steps_cycleorder.columns = np.arange(1, len(N_steps_cycleorder.columns)+1)
    return AEE_kJkg_cycleorder, N_steps_cycleorder, sorted_model_output

def reshape_df(AEE_kJkg_cycleorder, N_steps_cycleorder, sorted_model_output):
    """Reshape dataframe to wide."""
    df = AEE_kJkg_cycleorder.melt(ignore_index=False)
    df.columns = ['cycle_day', 'AEE_kJkg']

    ordered_df = {'subject':[], 'phase':[],'study_day':[], 'cycle_day':[], 'day_from_LH':[], 'AEE_kJkg':[], 'TEE_W':[], 'N_steps':[]}
    for subject in df.index.unique():
        sorted_days = list(sorted_model_output[subject]['TEE_J'].keys())
        for idx in range(len(sorted_days)):
            cycle_day = idx + 1
            
            # get menstrual cycle phase labels (menstrual, follicular, ovulation, luteal)
            phase_label = sorted_model_output[subject]['phase'][idx]
            
            # get timing from LH
            if len(sorted_model_output[subject]['time_from_LH']) == 0:
                day_from_LH = np.nan
            else:
                day_from_LH = float(sorted_model_output[subject]['time_from_LH'][idx])
                
            # get study day visit label
            study_day = sorted_model_output[subject]['study_day'][idx]
            
            # get AEE and number of steps
            AEE_kJkg = df['AEE_kJkg'][(df['cycle_day'] == idx+1) & (df.index == subject)].values[0]
            N_steps = N_steps_cycleorder[idx+1][N_steps_cycleorder.index == subject].values[0]

            # calculate mean TEE per step
            TEE_W = np.mean(sorted_model_output[subject]['TEE_W'][cycle_day])
            
            ordered_df['subject'].append(subject)
            ordered_df['phase'].append(phase_label)
            ordered_df['study_day'].append(study_day)
            ordered_df['day_from_LH'].append(day_from_LH)
            ordered_df['AEE_kJkg'].append(AEE_kJkg)
            ordered_df['cycle_day'].append(cycle_day)
            ordered_df['TEE_W'].append(TEE_W)
            ordered_df['N_steps'].append(N_steps)
            
    df_cycleorder = pd.DataFrame.from_dict(ordered_df, orient='columns')
    return df_cycleorder
    

def normality_test(df_cycleorder):
    # assess if df_cycleorder data are normal using Shapiro-Wilk test
    for phase in df_cycleorder['phase'].unique():
        data = df_cycleorder['AEE_kJkg'][df_cycleorder['phase'] == phase]
        stat, p = stats.shapiro(data)
        print(f'Shapiro-Wilk test for {phase}: stat={stat}, p={p}')
        if p > 0.05:
            print(f'Data for {phase} is normally distributed (fail to reject H0)\n')
        else:
            print(f'Data for {phase} is not normally distributed (reject H0)\n')


     