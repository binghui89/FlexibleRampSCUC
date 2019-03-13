# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:26:36 2018

@author: ksedzro
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 17 09:15:54 2018

@author: ksedzro
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:09:02 2018

@author: ksedzro
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:10:21 2018

@author: ksedzro
"""

########################################################################################################
# a basic (thermal) unit commitment model, drawn from:                                                 #
# A Computationally Efficient Mixed-Integer Linear Formulation for the Thermal Unit Commitment Problem #
# Miguel Carrion and Jose M. Arroyo                                                                    #
# IEEE Transactions on Power Systems, Volume 21, Number 3, August 2006. 
# Model with bus-wise curtailment and reserve/ramp shortages                               #
########################################################################################################
import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pandas import DataFrame
import time
from IPython import embed as IP

class MyDataFrame(DataFrame):
    def to_dict_2d(self, tol=None):
        dict_df = dict()
        if tol:
            for i in self.index:
                for j in self.columns:
                    v = self.loc[i, j]
                    if abs(v) >= tol:
                        dict_df[i,j] = v
        else:
            for i in self.index:
                for j in self.columns:
                    dict_df[i,j] = self.loc[i, j]
        return dict_df

class Network(object):
    def __init__(self, csvbus, csvbranch, csvptdf, csvgen, csvmarginalcost, csvblockmarginalcost, csvblockoutputlimit):
        self.df_bus              = pd.read_csv(csvbus,               index_col=['BUS_ID'])
        self.df_branch           = pd.read_csv(csvbranch,            index_col=['BR_ID'])
        self.df_ptdf             = pd.read_csv(csvptdf,              index_col=0)
        self.df_gen              = pd.read_csv(csvgen,               index_col=0)
        self.df_margcost         = pd.read_csv(csvmarginalcost,      index_col=0)
        self.df_blockmargcost    = pd.read_csv(csvblockmarginalcost, index_col=0)
        self.df_blockoutputlimit = pd.read_csv(csvblockoutputlimit,  index_col=0)
        self.enforce_kv_level(230) # Enforce the 230 kV limits
        self.update_gen_param() # Add start-up/shut-down costs and min on/offline hours

    def enforce_kv_level(self, kv_level):
        bus_kVlevel_set = set(self.df_bus[self.df_bus['BASEKV']>=kv_level].index)
        branch_kVlevel_set = {
            i for i in self.df_branch.index
            if self.df_branch.loc[i,'F_BUS'] in bus_kVlevel_set 
            and self.df_branch.loc[i,'T_BUS'] in bus_kVlevel_set
        }
        self.set_valid_branch = branch_kVlevel_set
        self.df_ptdf = self.df_ptdf.loc[self.set_valid_branch,:].copy()

    def update_gen_param(self):
        # This function add start-up/shut-down costs and min on/offline hours
        # This iteration-based way is not the most Pythonic, but it gets better readability
        for i, row in self.df_gen.iterrows():
            cap    = self.df_gen.loc[i, 'PMAX']
            s_cost = self.df_gen.loc[i, 'STARTUP']
            tmin   = self.df_gen.loc[i, 'MINIMUM_UP_TIME']
            t0     = self.df_gen.loc[i, 'GEN_STATUS']
            if i.startswith('coal'):
                if cap <= 300:
                    # Subcritical steam cycle
                    s_cost = 16.28*cap
                    tmin = 4 # Min on/offline time
                    t0 = tmin # Initial online time
                else:
                    # Supercritical steam cycle
                    s_cost = 29.44*cap
                    tmin = 12
                    t0 = tmin
            elif i.startswith('ng'):
                if cap <= 50:
                    # Small aeroderivative turbines
                    s_cost = 8.23*cap
                    tmin = 1
                    t0 = tmin
                elif cap <= 100:
                    # Small aeroderivative turbines
                    s_cost = 8.23*cap
                    tmin = 3
                    t0 = tmin
                elif cap <= 450:
                    # Heavy-duty GT
                    s_cost = 1.70*cap
                    tmin = 5
                    t0 = tmin
                else:
                    # CCGT
                    s_cost = 1.74*cap
                    tmin = 8
                    t0 = tmin
            elif i.startswith('nuc'):
                # Supercritical steam cycle
                s_cost = 29.44*cap
                tmin = 100
                t0 = 1 # Nuclear units never go offline

            self.df_gen.loc[i, 'STARTUP']  = s_cost
            self.df_gen.loc[i, 'SHUTDOWN'] = s_cost
            self.df_gen.loc[i, 'MINIMUM_UP_TIME']   = tmin
            self.df_gen.loc[i, 'MINIMUM_DOWN_TIME'] = tmin
            self.df_gen.loc[i, 'GEN_STATUS'] = t0 # All but nuclear units are free to be go offline

        self.df_gen['STARTUP_RAMP']  = self.df_gen[['STARTUP_RAMP','PMIN']].max(axis=1)
        self.df_gen['SHUTDOWN_RAMP'] = self.df_gen[['SHUTDOWN_RAMP','PMIN']].max(axis=1)

        # Assume renewable sources cost nothing to start
        self.df_gen.loc[self.df_gen['GEN_TYPE']=='Renewable', 'STARTUP'] = 0

        # Update 1 hour ramp rates
        self.df_gen['RAMP_60'] = self.df_gen['RAMP_10']*6

    def return_dict(self, attr, tol=None):
        # This method is a generic function that converts a Pandas dataframe 
        # into a dictionary.
        df = getattr(self, attr)
        dict_df = dict()
        if tol:
            for i in df.index:
                for j in df.columns:
                    v = df.loc[i, j]
                    if abs(v) >= tol:
                        dict_df[i,j] = v
        else:
            for i in df.index:
                for j in df.columns:
                    dict_df[i,j] = df.loc[i, j]
        return dict_df

    def return_dict_ptdf(self, tol=None):
        # Find out all shift factors that are greater than tolerance, which 
        # defaults to 1e-5
        dict_ptdf = self.return_dict('df_ptdf', tol)
        return dict_ptdf

    def return_dict_blockmargcost(self):
        dict_blockmargcost = self.return_dict('df_blockmargcost')
        return dict_blockmargcost

    def return_dict_blockoutputlimit(self):
        dict_blockoutputlimit = self.return_dict('df_blockoutputlimit')
        return dict_blockoutputlimit

#=======================================================#
# INPUT DATA                                            #
#=======================================================#
def da_input():
    data_path = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/'

    # Load network data
    branch_df = pd.read_csv(data_path+'branch.csv',index_col=['BR_ID'])
    bus_df    = pd.read_csv(data_path+'bus.csv',index_col=['BUS_ID'])
    bus_df['VOLL'] = 9000 # Note that we are using ERCOT's VOLL value, 9000 $/MWh
    kV_level = 230
    bus_kVlevel_set    = list(bus_df[bus_df['BASEKV']>=kV_level].index)
    branch_kVlevel_set = [
        i for i in branch_df.index
        if branch_df.loc[i,'F_BUS'] in bus_kVlevel_set 
        and branch_df.loc[i,'T_BUS'] in bus_kVlevel_set
    ]
    valid_id = branch_kVlevel_set
    ptdf_df = pd.read_csv(data_path+'ptdf.csv',index_col=0) # for case 118
    ptdf_df = ptdf_df.loc[valid_id,:].copy()

    # Read generator data
    gen_df = pd.read_csv(data_path+'generator_data_plexos_withRT.csv',index_col=0)
    gen_df['RAMP_60'] = gen_df['RAMP_10']*6 # 60 minutes ramp rates
    gen_df.loc[gen_df['GEN_TYPE']=='Renewable', 'STARTUP'] = 0

    # This iteration-based way is not the most Pythonic, but it gets better readability
    for i, row in gen_df.iterrows():
        cap    = gen_df.loc[i, 'PMAX']
        s_cost = gen_df.loc[i, 'STARTUP']
        tmin   = gen_df.loc[i, 'MINIMUM_UP_TIME']
        t0     = gen_df.loc[i, 'GEN_STATUS']
        if i.startswith('coal'):
            if cap <= 300:
                # Subcritical steam cycle
                s_cost = 16.28*cap
                tmin = 4 # Min on/offline time
                t0 = tmin # Initial online time
            else:
                # Supercritical steam cycle
                s_cost = 29.44*cap
                tmin = 12
                t0 = tmin
        elif i.startswith('ng'):
            if cap <= 50:
                # Small aeroderivative turbines
                s_cost = 8.23*cap
                tmin = 1
                t0 = tmin
            elif cap <= 100:
                # Small aeroderivative turbines
                s_cost = 8.23*cap
                tmin = 3
                t0 = tmin
            elif cap <= 450:
                # Heavy-duty GT
                s_cost = 1.70*cap
                tmin = 5
                t0 = tmin
            else:
                # CCGT
                s_cost = 1.74*cap
                tmin = 8
                t0 = tmin
        elif i.startswith('nuc'):
            # Supercritical steam cycle
            s_cost = 29.44*cap
            tmin = 100
            t0 = 1 # Nuclear units never go offline

        gen_df.loc[i, 'STARTUP']  = s_cost
        gen_df.loc[i, 'SHUTDOWN'] = s_cost
        gen_df.loc[i, 'MINIMUM_UP_TIME']   = tmin
        gen_df.loc[i, 'MINIMUM_DOWN_TIME'] = tmin
        gen_df.loc[i, 'GEN_STATUS'] = t0 # All but nuclear units are free to be go offline

    gen_df['STARTUP_RAMP']  = gen_df[['STARTUP_RAMP','PMIN']].max(axis=1)
    gen_df['SHUTDOWN_RAMP'] = gen_df[['SHUTDOWN_RAMP','PMIN']].max(axis=1)

    wind_generator_names  =  [x for x in gen_df.index if x.startswith('wind')]

    genth_df = pd.DataFrame(gen_df[gen_df['GEN_TYPE']=='Thermal'])

    # Read cost data
    margcost_df         = pd.read_csv(data_path+'marginalcost.csv',      index_col=0)
    blockmargcost_df    = pd.read_csv(data_path+'blockmarginalcost.csv', index_col=0)
    blockoutputlimit_df = pd.read_csv(data_path+'blockoutputlimit.csv',  index_col=0)

    # Read power forecast of renewable sources
    genfor_df = pd.read_csv(data_path+'generator.csv',index_col=0)
    genforren_df = pd.DataFrame()
    genforren_df = genfor_df.loc[:,gen_df[gen_df['GEN_TYPE']!='Thermal'].index]
    genforren_df.fillna(0, inplace=True)

    # Read load
    load_df = pd.read_csv(data_path+'loads.csv',index_col=0)

    ##################################################
    # Create dictionaries

    load_dict = dict()
    load_s_df = load_df[load_df.columns.difference(['LOAD'])].copy()
    columns = load_s_df.columns
    for i, t in load_s_df.iterrows():
        for col in columns:
            load_dict[(col, i)] = t[col]
    
    print('Start with the ptdf dictionary')

    # ptdf_dict = ptdf_df.to_dict('index') # should be indexed ptdf_dict[l][b]
    ptdf_dict = dict()
    for i in ptdf_df.index: # Branch
        for j in ptdf_df.columns: # Bus
            ptdf_dict[i, j] = ptdf_df.loc[i, j]
    print('Done with the ptdf dictionary')


    genforren_dict = dict()
    columns = genforren_df.columns
    for i, t in genforren_df.iterrows():
        for col in columns:
            genforren_dict[(col, i+1)] = t[col]

    print('Done with the forecast dictionary')

    blockmargcost_dict = dict()    
    columns = blockmargcost_df.columns
    for i, t in blockmargcost_df.iterrows():
        for col in columns:
            #print (i,t)
            blockmargcost_dict[(i,col)] = t[col]        

    print('Done with the block marginal cost dictionary')

    blockoutputlimit_dict = dict()
    columns = blockoutputlimit_df.columns
    for i, t in blockoutputlimit_df.iterrows():
        for col in columns:
            #print (i,t)
            blockoutputlimit_dict[(i,col)] = t[col]
    print('Done with all dictionaries')
    #================================================

    # Reserve Parameters
    ReserveFactor = 0.1
    RegulatingReserveFactor = 0.05

    return (
        gen_df,
        genth_df,
        wind_generator_names,
        blockmargcost_df,
        margcost_df,
        load_df,
        load_s_df,
        bus_df,
        branch_df,
        valid_id,
        blockoutputlimit_dict,
        blockmargcost_dict,
        load_dict,
        genforren_dict,
        ptdf_dict,
        ReserveFactor,
        RegulatingReserveFactor,
    )

# Minimum and maximum generation levels, for each thermal generator in MW.
# could easily be specified on a per-time period basis, but are not currently.
def maximum_power_output_validator(m, v, g):
   return v >= value(m.MinimumPowerOutput[g])

# Limits for time periods in which generators are brought on or off-line. 
# Must not be less than the generator minimum output. 
def at_least_generator_minimum_output_validator(m, v, g):
   return v >= m.MinimumPowerOutput[g]

# Unit on state at t=0 (initial condition), the value cannot be 0, by definition.
# if positive, the number of hours prior to (and including) t=0 that the unit has been on.
# if negative, the number of hours prior to (and including) t=0 that the unit has been off.
def t0_state_nonzero_validator(m, v, g):
    return v != 0

def t0_unit_on_rule(m, g):
    return int( value(m.UnitOnT0State[g]) >= 1 )

# The number of time periods that a generator must initally on-line (off-line) 
# due to its minimum up time (down time) constraint.
def initial_time_periods_online_rule(m, g):
   if not value(m.UnitOnT0[g]):
      return 0
   else:
      return min(
          value(m.NumTimePeriods),
          max(0, value(m.MinimumUpTime[g]) - value(m.UnitOnT0State[g]))
      )

def initial_time_periods_offline_rule(m, g):
   if value(m.UnitOnT0[g]):
      return 0
   else:
      return min(
          value(m.NumTimePeriods), 
          max(0, value(m.MinimumDownTime[g]) + value(m.UnitOnT0State[g]))
      ) # m.UnitOnT0State is negative if unit is off

# Spinning and Regulating Reserves requirements
def _reserve_requirement_rule(m, t):
    return m.ReserveFactor*sum(value(m.BusDemand[b,t]) for b in m.LoadBuses)

def _regulating_requirement_rule(m, t):
    return m.RegulatingReserveFactor*sum(value(m.BusDemand[b,t]) for b in m.LoadBuses)


#############################################
# supply-demand constraints
#############################################
# meet the demand at each time period.
# encodes Constraint 2 in Carrion and Arroyo.
def definition_hourly_curtailment_rule(m, t):
   return m.Curtailment[t] == sum(m.BusCurtailment[b, t] for b in m.LoadBuses)

def production_equals_demand_rule(m, t):
   return sum(m.PowerGenerated[g, t] for g in m.AllGenerators) == m.Demand[t] - m.Curtailment[t]

#############################################
# generation limit and ramping constraints
#############################################

# Enforce the generator power output limits on a per-period basis, the maximum 
# power available at any given time period is dynamic, bounded from above by the
# maximum generator output.
# The following three constraints encode Constraints 16 and 17 defined in 
# Carrion and Arroyo.

# NOTE: The expression below is what we really want - however, due to a pyomo 
# bug, we have to split it into two constraints:
# m.MinimumPowerOutput[g] * m.UnitOn[g, t] <= m.PowerGenerated[g,t] <= m.MaximumPowerAvailable[g, t]
# When fixed, merge back parts "a" and "b", leaving two constraints.

def enforce_generator_output_limits_rule_part_a(m, g, t):
   return m.MinimumPowerOutput[g]*m.UnitOn[g, t] <= m.PowerGenerated[g,t]

def enforce_generator_output_limits_rule_part_b(m, g, t):
   return m.PowerGenerated[g,t] <= m.MaximumPowerAvailable[g, t]

def enforce_generator_output_limits_rule_part_c(m, g, t):
   return m.MaximumPowerAvailable[g, t] <= m.MaximumPowerOutput[g]*m.UnitOn[g, t]

# Maximum available power of non-thermal units less than forecast
def enforce_renewable_generator_output_limits_rule(m, g, t):
   return  m.MaximumPowerAvailable[g, t]<= m.PowerForecast[g,t]

# Power generation of thermal units by block
def enforce_generator_block_output_rule(m, g, t):
    return m.PowerGenerated[g, t] == (
        # m.UnitOn[g,t]*margcost_df.loc[g,'Pmax0'] + 
        m.UnitOn[g,t]*m.BlockSize0[g] + 
        sum(
           m.BlockPowerGenerated[g,k,t]
           for k in m.Blocks
       )
    )

def enforce_generator_block_output_limit_rule(m, g, k, t):
   return m.BlockPowerGenerated[g,k,t] <= m.BlockSize[g,k]


# impose upper bounds on the maximum power available for each generator in each time period, 
# based on standard and start-up ramp limits.

# the following constraint encodes Constraint 18 defined in Carrion and Arroyo.

def enforce_max_available_ramp_up_rates_rule(m, g, t):
    # 4 cases, split by (t-1, t) unit status (RHS is defined as the delta from 
    # m.PowerGenerated[g, t-1])
    # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate 
    #                                    upper bound due to unit being off) 
    # (0, 1) - unit switching on:  RHS = startup ramp limit 
    # (1, 0) - unit switching off: RHS = standard ramp limit minus startup ramp 
    #                                    limit plus maximum power generated (
    #                                    degenerate upper bound due to unit off)
    # (1, 1) - unit staying on:    RHS = standard ramp limit
    if t == 1: # Maybe we can use m.TimePeriod.first() here
        return m.MaximumPowerAvailable[g, t] <= (
            m.PowerGeneratedT0[g] 
            + m.NominalRampUpLimit[g] * m.UnitOnT0[g] 
            + m.StartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOnT0[g]) 
            + m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
        )
    else:
        return m.MaximumPowerAvailable[g, t] <= (
            m.PowerGenerated[g, t-1] 
            + m.NominalRampUpLimit[g] * m.UnitOn[g, t-1] 
            + m.StartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, t-1]) 
            + m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
        )

# the following constraint encodes Constraint 19 defined in Carrion and Arroyo.

def enforce_max_available_ramp_down_rates_rule(m, g, t):
    # 4 cases, split by (t, t+1) unit status
    # (0, 0) - unit staying off:   RHS = 0 (degenerate upper bound)
    # (0, 1) - unit switching on:  RHS = maximum generator output minus shutdown 
    #                                    ramp limit (degenerate upper bound) 
    #                                    - this is the strangest case.
    # (1, 0) - unit switching off: RHS = shutdown ramp limit
    # (1, 1) - unit staying on:    RHS = maximum generator output (degenerate 
    #                                    upper bound)
    if t == value(m.NumTimePeriods): # Maybe we can use m.TimePeriod.last() here
        return Constraint.Skip
    else:
        return m.MaximumPowerAvailable[g, t] <= (
            m.MaximumPowerOutput[g] * m.UnitOn[g, t+1] 
            + m.ShutdownRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, t+1])
        )

# the following constraint encodes Constraint 20 defined in Carrion and Arroyo.

def enforce_ramp_down_limits_rule(m, g, t):
    # 4 cases, split by (t-1, t) unit status: 
    # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate 
    #                              upper bound)
    # (0, 1) - unit switching on:  RHS = standard ramp-down limit minus 
    #                                    shutdown ramp limit plus maximum 
    #                                    generator output - this is the 
    #                                    strangest case.
    # (1, 0) - unit switching off: RHS = shutdown ramp limit 
    # (1, 1) - unit staying on:    RHS = standard ramp-down limit 
    if t == 1:
        return m.PowerGeneratedT0[g] - m.PowerGenerated[g, t] <= (
            m.NominalRampDownLimit[g] * m.UnitOn[g, t] 
            + m.ShutdownRampLimit[g] * (m.UnitOnT0[g] - m.UnitOn[g, t]) 
            + m.MaximumPowerOutput[g] * (1 - m.UnitOnT0[g])
        )
    else:
        return (
            m.PowerGenerated[g, t-1] 
            - m.PowerGenerated[g, t] 
            - m.RegulatingReserveDnAvailable[g,t]
        ) <= (
            m.NominalRampDownLimit[g] * m.UnitOn[g, t] 
            + m.ShutdownRampLimit[g] * (m.UnitOn[g, t-1] - m.UnitOn[g, t]) 
            + m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t-1])
        )

#############################################
# Constraints for line capacity limits
#############################################

def line_flow_rule(m, l, t):
    # This is an expression of the power flow on bus b in time t, defined here
    # to save time.
    return sum(
        # ptdf_dict[l][m.GenBuses[g]]*m.PowerGenerated[g,t] 
        m.PTDF[l, m.GenBuses[g]]*m.PowerGenerated[g,t] 
        for g in m.AllGenerators
    ) - sum(
        # ptdf_dict[l][b]*(m.BusDemand[b,t] - m.BusCurtailment[b,t]) 
        m.PTDF[l, b]*(m.BusDemand[b,t] - m.BusCurtailment[b,t]) 
        for b in m.LoadBuses
    )

def enforce_line_capacity_limits_rule_a(m, l, t):
    return m.LineFlow[l, t] <= m.LineLimits[l]

def enforce_line_capacity_limits_rule_b(m, l, t):
    return m.LineFlow[l, t] >= -m.LineLimits[l]


#############################################
# Up-time constraints #
#############################################

# Constraint due to initial conditions.
def enforce_up_time_constraints_initial(m, g):
    if value(m.InitialTimePeriodsOnLine[g]) is 0:
        return Constraint.Skip
    return sum(
        (1 - m.UnitOn[g, t]) 
        for g in m.ThermalGenerators 
        for t in m.TimePeriods if t <= value(m.InitialTimePeriodsOnLine[g])
    ) == 0.0

# Constraint for each time period after that not involving the initial condition.
def enforce_up_time_constraints_subsequent(m, g, t):
    if t <= value(m.InitialTimePeriodsOnLine[g]):
        # handled by the EnforceUpTimeConstraintInitial constraint.
        return Constraint.Skip
    elif t <= (value(m.NumTimePeriods) - value(m.MinimumUpTime[g]) + 1): # Maybe only use one value
        # The right-hand side terms below are only positive if the unit was off 
        # in time (t - 1) but on in time t, and the value is the minimum number 
        # of subsequent consecutive time periods that the unit must be on.
        if t is 1:
            return sum(
                m.UnitOn[g, n] 
                for n in m.TimePeriods 
                if n >= t and n <= t + value(m.MinimumUpTime[g]) - 1
            ) >= m.MinimumUpTime[g] * (m.UnitOn[g, t] - m.UnitOnT0[g])
        else:
            return sum(
                m.UnitOn[g, n] 
                for n in m.TimePeriods 
                if n >= t and n <= t + value(m.MinimumUpTime[g]) - 1
            ) >= m.MinimumUpTime[g] * (m.UnitOn[g, t] - m.UnitOn[g, t-1])
    else:
        # Handle the final (MinimumUpTime[g] - 1) time periods - if a unit is 
        # started up in this interval, it must remain on-line until the end of 
        # the time span.
        if t == 1: # can happen when small time horizons are specified
            return sum(
                m.UnitOn[g, n] - (m.UnitOn[g, t] - m.UnitOnT0[g])
                for n in m.TimePeriods if n >= t
            ) >= 0.0
        else:
            return sum(
                m.UnitOn[g, n] - (m.UnitOn[g, t] - m.UnitOn[g, t-1]) 
                for n in m.TimePeriods if n >= t
            ) >= 0.0

#############################################
# Down-time constraints
#############################################

# constraint due to initial conditions.
def enforce_down_time_constraints_initial(m, g):
   if value(m.InitialTimePeriodsOffLine[g]) is 0: 
      return Constraint.Skip
   return sum(
       m.UnitOn[g, t] 
       for g in m.ThermalGenerators 
       for t in m.TimePeriods if t <= value(m.InitialTimePeriodsOffLine[g])
   ) == 0.0


# constraint for each time period after that not involving the initial condition.
def enforce_down_time_constraints_subsequent(m, g, t):
    if t <= value(m.InitialTimePeriodsOffLine[g]):
        # handled by the EnforceDownTimeConstraintInitial constraint.
        return Constraint.Skip
    elif t <= (value(m.NumTimePeriods) - value(m.MinimumDownTime[g]) + 1):
        # The right-hand side terms below are only positive if the unit was on 
        # in time (t - 1) but on in time, and the value is the minimum number of 
        # subsequent consecutive time periods that the unit must be on.
        if t is 1:
            return sum(
                1 - m.UnitOn[g, n]
                for n in m.TimePeriods 
                if n >= t and n <= t + value(m.MinimumDownTime[g]) - 1
            ) >= m.MinimumDownTime[g] * (m.UnitOnT0[g] - m.UnitOn[g, t])
        else:
            return sum(
                1 - m.UnitOn[g, n] 
                for n in m.TimePeriods 
                if n >= t and n <= t + value(m.MinimumDownTime[g]) - 1
            ) >= m.MinimumDownTime[g] * (m.UnitOn[g, t-1] - m.UnitOn[g, t])
    else:
        # handle the final (MinimumDownTime[g] - 1) time periods - if a unit is 
        # shut down in this interval, it must remain off-line until the end of 
        # the time span.
        if t == 1: # can happen when small time horizons are specified
            return sum(
                (1 - m.UnitOn[g, n]) - (m.UnitOnT0[g] - m.UnitOn[g, t])
                for n in m.TimePeriods if n >= t
            ) >= 0.0
        else:
            return sum(
                (1 - m.UnitOn[g, n]) - (m.UnitOn[g, t-1] - m.UnitOn[g, t]) 
                for n in m.TimePeriods if n >= t
            ) >= 0.0


#############################################
# Regulating and spinning availability
#############################################

def reserve_up_by_maximum_available_power_thermal_rule(m, g, t):
    return (
        m.PowerGenerated[g, t]
        + m.RegulatingReserveUpAvailable[g, t]
        + m.SpinningReserveUpAvailable[g, t]
        - m.MaximumPowerAvailable[g, t]
    ) <= 0

def reserve_dn_by_maximum_available_power_thermal_rule(m, g, t):
    return (
        m.PowerGenerated[g, t]
        - m.RegulatingReserveDnAvailable[g, t]
        - m.MinimumPowerOutput[g] * m.UnitOn[g, t]
    ) >= 0

def reserve_up_by_ramp_thermal_rule(m, g, t):
    # Note ERCOT regulates that reserves must be able to ramp to awarded level
    # within 10 minutes.
    return (
        m.RegulatingReserveUpAvailable[g, t]
        + m.SpinningReserveUpAvailable[g, t]
        - 10/60 * m.NominalRampUpLimit[g] * m.UnitOn[g, t]
    ) <= 0

def reserve_dn_by_ramp_thermal_rule(m, g, t):
    # Note ERCOT regulates that reserves must be able to ramp to awarded level
    # within 10 minutes.
    return (
        m.RegulatingReserveDnAvailable[g, t]
        - 10/60 * m.NominalRampDownLimit[g] * m.UnitOn[g, t]
    ) <= 0    

# model.reserve_up_by_ramp_thermal_constraint = Constraint(
#     model.ThermalGenerators, model.TimePeriods,
#     rule=reserve_up_by_ramp_thermal_rule
# )
# model.reserve_dn_by_ramp_thermal_constraint = Constraint(
#     model.ThermalGenerators, model.TimePeriods,
#     rule=reserve_dn_by_ramp_thermal_rule
# )

# Spinning reserve requirements
def enforce_spinning_reserve_requirement_rule(m,  t):
    return sum(
        m.SpinningReserveUpAvailable[g,t] for g in m.ThermalGenerators
    ) + m.SpinningReserveUpShortage[t] - m.SpinningReserveRequirement[t] == 0


# Regulating reserve requirements
def enforce_regulating_up_reserve_requirement_rule(m, t):
     return sum(
         m.RegulatingReserveUpAvailable[g,t] for g in m.ThermalGenerators
     ) + m.RegulatingReserveUpShortage[t] - m.RegulatingReserveRequirement[t] == 0
 
def enforce_regulating_down_reserve_requirement_rule(m, t):
     return sum(
         m.RegulatingReserveDnAvailable[g,t] for g in m.ThermalGenerators
     ) + m.RegulatingReserveDnShortage[t] - m.RegulatingReserveRequirement[t] == 0


#############################################
# constraints for computing cost components #
#############################################

# Production cost, per gen per time slice
def production_cost_function(m, g, t):
    return m.ProductionCost[g,t] == (
        # m.UnitOn[g,t]*margcost_df.loc[g,'nlcost']
        m.UnitOn[g,t]*m.BlockMarginalCost0[g]
        + sum(
            value(m.BlockMarginalCost[g,k])*(m.BlockPowerGenerated[g,k,t]) 
            for k in m.Blocks
        )
    )

# Compute the total production costs, across all generators and time periods.
def compute_total_production_cost_rule(m):
   return m.TotalProductionCost == sum(
       m.ProductionCost[g, t] 
       for g in m.ThermalGenerators 
       for t in m.TimePeriods
   )

# Compute the per-generator, per-time period shut-down and start-up costs.
def compute_shutdown_costs_rule(m, g, t):
   if t is 1: # Maybe replace with .first()
      return m.ShutdownCost[g, t] >= m.ShutdownCostCoefficient[g] * (m.UnitOnT0[g] - m.UnitOn[g, t])
   else:
      return m.ShutdownCost[g, t] >= m.ShutdownCostCoefficient[g] * (m.UnitOn[g, t-1] - m.UnitOn[g, t])

def compute_startup_costs_rule(m, g, t):
   if t is 1: # Maybe replace with .first()
      return m.StartupCost[g, t] >= m.StartupCostCoefficient[g] * (-m.UnitOnT0[g] + m.UnitOn[g, t])
   else:
      return m.StartupCost[g, t] >= m.StartupCostCoefficient[g] * (-m.UnitOn[g, t-1] + m.UnitOn[g, t])

# Compute the total startup and shutdown costs, across all generators and time periods.
def compute_total_fixed_cost_rule(m):
   return m.TotalFixedCost == sum(
       m.StartupCost[g, t] + m.ShutdownCost[g, t]
       for g in m.ThermalGenerators 
       for t in m.TimePeriods
   )

# Compute the total load curtailment cost
def compute_total_curtailment_cost_rule(m):
    return m.TotalCurtailmentCost == sum(
        m.BusVOLL[b] * m.BusCurtailment[b,t]
        for b in m.LoadBuses 
        for t in m.TimePeriods
    )

# Compute the total reserve shortage cost
def compute_total_reserve_shortage_cost_rule(m):
    return m.TotalReserveShortageCost == sum(
        m.SpinningReserveUpShortage[t] * 2000
        + m.RegulatingReserveUpShortage[t] * 5500
        + m.RegulatingReserveDnShortage[t] * 5500
        for t in m.TimePeriods
    )

# Objectives
def total_cost_objective_rule(m):
   return (
       m.TotalProductionCost
       + m.TotalFixedCost
       + m.TotalCurtailmentCost
       + m.TotalReserveShortageCost
   )

def create_model(
    network,
    df_busload, # Only bus load, first dimension time starts from 1, no total load
    df_genfor_nonthermal, # Only generation from nonthermal gens, first dim time starts from 1
    ReserveFactor,
    RegulatingReserveFactor,
):
    model = ConcreteModel()
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    """SETS"""
    ##########################################################
    # String indentifiers for the sets of different types of generators.
    i_thermal = (network.df_gen['GEN_TYPE']=='Thermal')
    model.AllGenerators        = Set(initialize=network.df_gen.index)
    model.ThermalGenerators    = Set(initialize=network.df_gen[i_thermal].index)
    model.NonThermalGenerators = Set(initialize=network.df_gen[~i_thermal].index)
    model.RenewableGenerators  = Set(initialize=network.df_gen[network.df_gen['GEN_TYPE']=='Renewable'].index)
    model.HydroGenerators      = Set(initialize=network.df_gen[network.df_gen['GEN_TYPE']=='Hydro'].index)
    model.WindGenerators       = Set(initialize=[i for i in network.df_gen.index if i.startswith('wind')])

    # Set of Generator Blocks Set.
    model.Blocks = Set(initialize = network.df_blockmargcost.columns)

    # String indentifiers for the set of load buses.
    # model.LoadBuses = Set(initialize=load_s_df.columns)
    model.LoadBuses = Set(initialize=df_busload.columns)
    model.Buses     = Set(initialize=network.df_bus.index)

    # String indentifiers for the set of branches.
    model.Branches         = Set(initialize=network.df_branch.index)
    model.EnforcedBranches = Set(initialize=network.set_valid_branch)

    # PTDF.
    # model.PTDF = Param(model.Buses, model.Branches, within=Reals, initialize=ptdf_dict)
    model.PTDF = Param(
        model.Branches, model.Buses,
        within=Reals,
        initialize=network.return_dict_ptdf(tol=None),
        default=0.0
    )

    # The number of time periods under consideration, in addition to the corresponding set.
    # model.NumTimePeriods = Param(within=PositiveIntegers, initialize=len(load_df.index))
    model.NumTimePeriods = Param(within=PositiveIntegers, initialize=len(df_busload.index))
    model.TimePeriods    = RangeSet(1, model.NumTimePeriods)

    """PARAMETERS"""
    ##########################################################
    # Buses indexed by all generators.
    model.GenBuses = Param(model.AllGenerators, initialize=network.df_gen['GEN_BUS'].to_dict())

    # Line capacity limits indexed by branches, units are MW.
    model.LineLimits = Param(
        model.Branches,
        # model.EnforcedBranches,
        within=NonNegativeReals,
        initialize = network.df_branch['RATE_A'].to_dict()
    )

    # The global system demand, for each time period. units are MW.
    # model.Demand = Param(
    #     model.TimePeriods,
    #     within=NonNegativeReals, initialize=load_df['LOAD'].to_dict(), mutable=True
    # )
    model.Demand = Param(
        model.TimePeriods,
        within=NonNegativeReals, 
        initialize=df_busload.sum(axis=1).to_dict(), 
        mutable=True
    )

    # The bus-by-bus demand and value of loss load, for each time period. units are MW and $/MW.
    # model.BusDemand = Param(
    #     model.LoadBuses, model.TimePeriods,
    #     within=NonNegativeReals, initialize=load_dict, mutable=True
    # )
    model.BusDemand = Param(
        model.LoadBuses, model.TimePeriods,
        within=NonNegativeReals,
        initialize=MyDataFrame(df_busload.T).to_dict_2d(), 
        mutable=True
    ) # Transpose because the first index of model.BusDemand is bus, not time.
    model.BusVOLL = Param(
        model.LoadBuses,
        within=NonNegativeReals,
        initialize=network.df_bus[ network.df_bus['PD']>0 ][ 'VOLL' ].to_dict()
    )

    # Power forecasts for renewables indexed by (gen, time)
    # model.PowerForecast = Param(
    #     model.NonThermalGenerators, model.TimePeriods,
    #     within=NonNegativeReals, initialize=genforren_dict, mutable=True
    # )
    model.PowerForecast = Param(
        model.NonThermalGenerators, model.TimePeriods,
        within=NonNegativeReals,
        initialize=MyDataFrame(df_genfor_nonthermal.T).to_dict_2d(),
        mutable=True
    )

    model.MinimumPowerOutput = Param(
        model.ThermalGenerators,
        # within=NonNegativeReals, initialize=genth_df['PMIN'].to_dict()
        within=NonNegativeReals,
        initialize=network.df_gen.loc[i_thermal, 'PMIN'].to_dict()
    )
    model.MaximumPowerOutput = Param(
        model.ThermalGenerators,
        # within=NonNegativeReals, initialize=genth_df['PMAX'].to_dict(),
        within=NonNegativeReals, 
        initialize=network.df_gen.loc[i_thermal, 'PMAX'].to_dict(),
        validate=maximum_power_output_validator
    )

    # Generator ramp up/down rates. units are MW/h.
    # Limits for normal time periods
    model.NominalRampUpLimit   = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=network.df_gen.loc[i_thermal, 'RAMP_60'].to_dict(),
    )
    model.NominalRampDownLimit = Param(
        model.ThermalGenerators,
        within=NonNegativeReals, 
        initialize=network.df_gen.loc[i_thermal, 'RAMP_60'].to_dict(),
    )

    # Limits for start-up/shut-down
    model.StartupRampLimit  = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=network.df_gen.loc[i_thermal, 'RAMP_60'].to_dict(),
        validate=at_least_generator_minimum_output_validator
    )
    model.ShutdownRampLimit = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=network.df_gen.loc[i_thermal, 'RAMP_60'].to_dict(),
        validate=at_least_generator_minimum_output_validator
    )

    # Min number of time periods that a gen must be on-line (off-line) once brought up (down).
    model.MinimumUpTime = Param(
        model.ThermalGenerators,
        within=NonNegativeIntegers,
        initialize=network.df_gen.loc[i_thermal, 'MINIMUM_UP_TIME'].to_dict(),
        mutable=True
    )
    model.MinimumDownTime = Param(
        model.ThermalGenerators,
        within=NonNegativeIntegers,
        initialize=network.df_gen.loc[i_thermal, 'MINIMUM_DOWN_TIME'].to_dict(),
        mutable=True
    )

    model.UnitOnT0State = Param(
        model.ThermalGenerators,
        within=Integers,
        initialize=network.df_gen.loc[i_thermal, 'GEN_STATUS'].to_dict(),
        validate=t0_state_nonzero_validator,
        mutable=True
    )

    model.UnitOnT0 = Param(
        model.ThermalGenerators,
        within=Binary,
        initialize=t0_unit_on_rule,
        mutable=True
    )

    model.InitialTimePeriodsOnLine = Param(
        model.ThermalGenerators,
        within=NonNegativeIntegers,
        initialize=initial_time_periods_online_rule,
        mutable=True
    )
    model.InitialTimePeriodsOffLine = Param(
        model.ThermalGenerators,
        within=NonNegativeIntegers,
        initialize=initial_time_periods_offline_rule,
        mutable=True
    )

    # Generator power output at t=0 (initial condition). units are MW.
    model.PowerGeneratedT0 = Param(
        model.AllGenerators, 
        within=NonNegativeReals,
        initialize=network.df_gen.loc[i_thermal, 'PMIN'].to_dict(),
        mutable=True
    )

    # Production cost coefficients (for the quadratic) 
    # a0=constant, a1=linear coefficient, a2=quadratic coefficient.

    # model.ProductionCostA0 = Param(
    #     model.ThermalGenerators,
    #     within=NonNegativeReals, initialize=gen_df['COST_0'].to_dict()
    # ) # units are $/hr (or whatever the time unit is).
    # model.ProductionCostA1 = Param(
    #     model.ThermalGenerators, 
    #     within=NonNegativeReals, initialize=margcost_df['1'].to_dict()
    # ) # units are $/MWhr.
    # model.ProductionCostA2 = Param(
    #     model.ThermalGenerators, 
    #     within=NonNegativeReals, initialize=gen_df['COST_2'].to_dict()
    # ) # units are $/(MWhr^2).
    model.BlockMarginalCost = Param(
        model.ThermalGenerators, model.Blocks, 
        within=NonNegativeReals,
        initialize=network.return_dict_blockmargcost()
    )
    # Number of cost function blockes indexed by (gen, block)
    model.BlockSize = Param(
        model.ThermalGenerators, model.Blocks,
        initialize=network.return_dict_blockoutputlimit()
    )

    model.BlockSize0 = Param(
        model.ThermalGenerators,
        initialize=network.df_margcost['Pmax0'].to_dict()
    )
    model.BlockMarginalCost0 = Param(
        model.ThermalGenerators,
        initialize=network.df_margcost['nlcost'].to_dict()
    )

    # Shutdown and startup cost for each generator, in the literature, these are 
    # often set to 0.
    model.ShutdownCostCoefficient = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=network.df_gen.loc[i_thermal, 'SHUTDOWN'].to_dict(),
    ) # units are $.
    model.StartupCostCoefficient = Param(
        model.ThermalGenerators, 
        within=NonNegativeReals,
        initialize=network.df_gen.loc[i_thermal, 'STARTUP'].to_dict(),
    ) # units are $.

    model.ReserveFactor = Param(
        within=Reals, initialize=ReserveFactor, default=0.0, mutable=True
    )
    model.RegulatingReserveFactor = Param(
        within=Reals, initialize=RegulatingReserveFactor, default=0.0, mutable=True
    )
    model.SpinningReserveRequirement = Param(
        model.TimePeriods, 
        within=NonNegativeReals, default=0.0, mutable=True,
        initialize=_reserve_requirement_rule
    )
    model.RegulatingReserveRequirement = Param(
        model.TimePeriods, 
        within=NonNegativeReals, default=0.0, mutable=True,
        initialize=_regulating_requirement_rule
    )


    """VARIABLES"""
    ##########################################################
    #  VARIABLE DEFINITION
    ##########################################################
    # Reserve variables
    model.RegulatingReserveUpAvailable = Var(
        model.ThermalGenerators, model.TimePeriods,
        within=NonNegativeReals, initialize=0.0
    )
    model.RegulatingReserveDnAvailable = Var(
        model.ThermalGenerators, model.TimePeriods,
        within=NonNegativeReals, initialize=0.0
    )
    model.SpinningReserveUpAvailable = Var(
        model.ThermalGenerators, model.TimePeriods, 
        within=NonNegativeReals, initialize=0.0
    )

    # Reserve shortages
    model.RegulatingReserveUpShortage = Var(
        model.TimePeriods,
        within=NonNegativeReals, initialize=0.0
    )
    model.RegulatingReserveDnShortage = Var(
        model.TimePeriods,
        within=NonNegativeReals, initialize=0.0
    )
    model.SpinningReserveUpShortage = Var(
        model.TimePeriods, 
        within=NonNegativeReals, initialize=0.0
    )

    # Generator related variables
    # Indicator variables for each generator, at each time period.
    model.UnitOn = Var(
        model.ThermalGenerators, model.TimePeriods,
        within=Binary, initialize=0
    )
    # Amount of power produced by each generator, at each time period.
    model.PowerGenerated = Var(
        model.AllGenerators, model.TimePeriods,
        within=NonNegativeReals, initialize=0.0
    )
    # Amount of power produced by each generator, in each block, at each time period.
    model.BlockPowerGenerated = Var(
        model.ThermalGenerators, model.Blocks, model.TimePeriods,
        within=NonNegativeReals, initialize=0.0
    )

    # Maximum power output for each generator, at each time period.
    model.MaximumPowerAvailable = Var(
        model.AllGenerators, model.TimePeriods, 
        within=NonNegativeReals, initialize=0.0
    )

    # Costs
    # Production cost associated with each generator, for each time period.
    model.ProductionCost = Var(
        model.ThermalGenerators, model.TimePeriods, 
        within=NonNegativeReals, initialize=0.0
    )
    # Cost over all generators, for all time periods.
    model.TotalProductionCost = Var(within=NonNegativeReals, initialize=0.0)

    # Startup and shutdown costs for each generator, each time period.
    model.StartupCost = Var(
        model.ThermalGenerators, model.TimePeriods, 
        within=NonNegativeReals, initialize=0.0
    )
    model.ShutdownCost = Var(
        model.ThermalGenerators, model.TimePeriods, 
        within=NonNegativeReals, initialize=0.0
    )
    model.TotalFixedCost = Var(within=NonNegativeReals, initialize=0.0)

    model.BusCurtailment = Var(
        model.LoadBuses,model.TimePeriods, 
        within=NonNegativeReals, initialize=0.0
    )
    model.Curtailment = Var(model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    model.TotalCurtailmentCost = Var(initialize=0.0, within=NonNegativeReals)
    model.TotalReserveShortageCost = Var(initialize=0.0, within=NonNegativeReals)

    """CONSTRAINTS"""
    ##########################################################
    # CONSTRAINTS
    ##########################################################
    model.DefineHourlyCurtailment = Constraint(
        model.TimePeriods, rule=definition_hourly_curtailment_rule
    )
    model.ProductionEqualsDemand = Constraint(
        model.TimePeriods, rule=production_equals_demand_rule
    )

    model.EnforceGeneratorOutputLimitsPartA = Constraint(
        model.ThermalGenerators, model.TimePeriods,
        rule=enforce_generator_output_limits_rule_part_a
    )
    model.EnforceGeneratorOutputLimitsPartB = Constraint(
        model.AllGenerators, model.TimePeriods, 
        rule=enforce_generator_output_limits_rule_part_b
    )
    model.EnforceGeneratorOutputLimitsPartC = Constraint(
        model.ThermalGenerators, model.TimePeriods, 
        rule=enforce_generator_output_limits_rule_part_c
    )

    model.EnforceRenewableOutputLimits = Constraint(
        model.NonThermalGenerators, model.TimePeriods, 
        rule=enforce_renewable_generator_output_limits_rule
    )

    model.EnforceGeneratorBlockOutput = Constraint(
        model.ThermalGenerators, model.TimePeriods, 
        rule=enforce_generator_block_output_rule
    )
    model.EnforceGeneratorBlockOutputLimit = Constraint(
        model.ThermalGenerators, model.Blocks, model.TimePeriods, 
        rule=enforce_generator_block_output_limit_rule
    )

    model.EnforceMaxAvailableRampUpRates = Constraint(
        model.ThermalGenerators, model.TimePeriods, 
        rule=enforce_max_available_ramp_up_rates_rule
    )

    model.EnforceMaxAvailableRampDownRates = Constraint(
        model.ThermalGenerators, model.TimePeriods, 
        rule=enforce_max_available_ramp_down_rates_rule
    )

    model.EnforceNominalRampDownLimits = Constraint(
        model.ThermalGenerators, model.TimePeriods, 
        rule=enforce_ramp_down_limits_rule
    )

    model.LineFlow = Expression(
        model.EnforcedBranches, model.TimePeriods,
        rule=line_flow_rule
    )
    model.EnforceLineCapacityLimitsA = Constraint(
        model.EnforcedBranches, model.TimePeriods, 
        rule=enforce_line_capacity_limits_rule_a
    )   
    model.EnforceLineCapacityLimitsB = Constraint(
        model.EnforcedBranches, model.TimePeriods, 
        rule=enforce_line_capacity_limits_rule_b
    )

    model.EnforceUpTimeConstraintsInitial = Constraint(
        model.ThermalGenerators, 
        rule=enforce_up_time_constraints_initial
    )

    model.EnforceUpTimeConstraintsSubsequent = Constraint(
        model.ThermalGenerators, model.TimePeriods, 
        rule=enforce_up_time_constraints_subsequent
    )

    model.EnforceDownTimeConstraintsInitial = Constraint(
        model.ThermalGenerators, 
        rule=enforce_down_time_constraints_initial
    )

    model.EnforceDownTimeConstraintsSubsequent = Constraint(
        model.ThermalGenerators, model.TimePeriods, 
        rule=enforce_down_time_constraints_subsequent
    )

    model.reserve_up_by_maximum_available_power_thermal_constraint = Constraint(
        model.ThermalGenerators, model.TimePeriods,
        rule=reserve_up_by_maximum_available_power_thermal_rule
    )
    model.reserve_dn_by_maximum_available_power_thermal_constraint = Constraint(
        model.ThermalGenerators, model.TimePeriods,
        rule=reserve_dn_by_maximum_available_power_thermal_rule
    )

    # model.reserve_up_by_ramp_thermal_constraint = Constraint(
    #     model.ThermalGenerators, model.TimePeriods,
    #     rule=reserve_up_by_ramp_thermal_rule
    # )
    # model.reserve_dn_by_ramp_thermal_constraint = Constraint(
    #     model.ThermalGenerators, model.TimePeriods,
    #     rule=reserve_dn_by_ramp_thermal_rule
    # )

    model.EnforceSpinningReserveUp = Constraint(
        model.TimePeriods, rule=enforce_spinning_reserve_requirement_rule
    )

    model.EnforceRegulatingUpReserveRequirements = Constraint(
        model.TimePeriods, rule=enforce_regulating_up_reserve_requirement_rule
    )
    model.EnforceRegulatingDnReserveRequirements = Constraint(
        model.TimePeriods, rule=enforce_regulating_down_reserve_requirement_rule
    )

    model.ComputeProductionCost = Constraint(
        model.ThermalGenerators, model.TimePeriods, 
        rule=production_cost_function
    )
    model.ComputeTotalProductionCost = Constraint(rule=compute_total_production_cost_rule)

    model.ComputeStartupCosts = Constraint(
        model.ThermalGenerators, model.TimePeriods, 
        rule=compute_startup_costs_rule
    )
    model.ComputeShutdownCosts = Constraint(
        model.ThermalGenerators, model.TimePeriods, 
        rule=compute_shutdown_costs_rule
    )
    model.ComputeTotalFixedCost = Constraint(rule=compute_total_fixed_cost_rule)
    model.ComputeTotalCurtailmentCost = Constraint(rule=compute_total_curtailment_cost_rule)
    model.ComputeTotalReserveShortageCost = Constraint(rule=compute_total_reserve_shortage_cost_rule)

    model.TotalCostObjective = Objective(
        rule=total_cost_objective_rule, sense=minimize
    )

    return model