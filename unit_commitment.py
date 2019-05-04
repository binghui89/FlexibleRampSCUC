import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pandas import DataFrame
from time import time
from matplotlib import pyplot as plt
from SCUC_RampConstraint_3 import create_model
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
        self.enforce_kv_level(0) # Enforce the 230 kV limits
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

def sigma_up_rule(m, g, t):
    t1 = m.TimePeriods.first()
    te = m.TimePeriods.last()
    Tsu = value(m.StartupTime[g])
    TU0 = value(m.UnitOnT0State[g])
    if t >= t1 + (Tsu - 1):
        return sum(
            m.UnitStartUp[g, t - i + 1]
            for i in range(1, Tsu + 1 )
        )
    elif (value(m.UnitOnT0[g]) == 1) and (TU0 < Tsu ) and (t <= t1 + Tsu - TU0 - 1):
        # Unit is starting up, and (t1 + Tsu - TU0 - 1) is the last starting up period
        # print g, t
        return 1
    else:
        return sum(
            m.UnitStartUp[g, t - i + 1]
            for i in range(1, (t-t1+1) + 1 ) # t-t1+1 is the number of intervals from t1 to t
        )

def sigma_dn_rule(m, g, t):
    t1  = m.TimePeriods.first()
    te  = m.TimePeriods.last()
    Tsd = value(m.ShutdownTime[g])
    if t <= te - Tsd:
        return sum(
            m.UnitShutDn[g, t + i]
            for i in range(1, Tsd + 1 )
        )
    elif t < te:
        return sum(
            m.UnitShutDn[g, t + i]
            for i in range(1, te - t + 1 )
        )
    else:
        # The last period, we don't know what's the value of the shut-down 
        # indicator of the next interval, so it's always 0.
        return 0

def sigma_power_times_up_rule(m, g, t):
    t1 = m.TimePeriods.first()
    te = m.TimePeriods.last()
    Tsu = value(m.StartupTime[g])
    TU0 = value(m.UnitOnT0State[g])
    Pmin = value(m.MinimumPowerOutput[g])
    if t >= t1 + (Tsu - 1):
        return sum(
            m.UnitStartUp[g, t - i + 1]*float(i)/Tsu*Pmin
            for i in range(1, Tsu + 1 )
        )
    elif (value(m.UnitOnT0[g]) == 1) and (TU0 < Tsu ) and (t <= t1 + Tsu - TU0 - 1):
        # Unit is starting up, and (t1 + STT - TU0 - 1) is the last starting up period
        # print g, t, float(t-t1+1+TU0)/Tsu*Pmin
        return float(t-t1+1+TU0)/Tsu*Pmin # t-t1+1+TU0 indicates which start-up period
    else:
        return sum(
            m.UnitStartUp[g, t - i + 1]*float(i)/Tsu*Pmin
            for i in range(1, (t-t1+1) + 1 ) # t-t1+1 is the number of intervals from t1 to t
        )

def sigma_power_times_dn_rule(m, g, t):
    t1  = m.TimePeriods.first()
    te  = m.TimePeriods.last()
    Tsd = value(m.ShutdownTime[g])
    Pmin = value(m.MinimumPowerOutput[g])
    if t <= te - Tsd:
        # float(Tsd-i+1)/Tsd*Pmin is the power level at the end of the ith 
        # shutting-down interval
        return sum(
            m.UnitShutDn[g, t + Tsd - i + 1]*float(Tsd-i+1)/Tsd*Pmin
            for i in range(1, Tsd + 1)
        )
    elif t < te:
        return sum(
            m.UnitShutDn[g, t + Tsd - i + 1]*float(Tsd-i+1)/Tsd*Pmin
            for i in range(t+Tsd+1-te, Tsd + 1)
        )
    else: # The last period
        return 0

def sigma_dn_initial_rule(m, g):
    t1  = m.TimePeriods.first()
    t0  = t1 - 1 # So t1 must be no less than 1
    Tsd = value(m.ShutdownTime[g])
    return sum(
        m.UnitShutDn[g, t0 + i]
        for i in range(1, Tsd + 1)
    )

def sigma_power_times_dn_initial_rule(m, g):
    t1  = m.TimePeriods.first()
    t0  = t1 - 1 # So t1 must be no less than 1
    Tsd = value(m.ShutdownTime[g])
    Pmin = value(m.MinimumPowerOutput[g])
    return sum(
        m.UnitShutDn[g, t0 + Tsd - i + 1]*float(Tsd-i+1)/Tsd*Pmin
        for i in range(1, Tsd + 1)
    )

def set_initial_shutdown_power_limits_rule(m):
    set_gen = set()
    for g in m.ThermalGenerators:
        if value(m.UnitOnT0State[g]) >= value(m.ShutdownTime[g]):
            set_gen.add(g)
    return set_gen

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
   return sum(m.PowerGenerated[g, t] for g in m.AllGenerators) + m.Curtailment[t] - m.OverCommit[t] == m.Demand[t]

#############################################
# generation limit and ramping constraints
#############################################

def init_set_fix_shutdown(m):
    '''
    This is to fix the shutting down indicators of thermal gens. Specifically,
    if at time 0, a thermal gen is on and satisfies the following condictions:
    1. Its power level is lower than Pmin.
    2. The unit has been on for longer than its start-up time.
    then this unit is shutting down, and the interval in which the shut-down
    indicator should be 1 is given below.
    '''
    set_gen_time = set()
    for g in m.ThermalGenerators:
        v0 = value(m.UnitOnT0[g])
        P0 = value(m.PowerGeneratedT0[g])
        Pmin = value(m.MinimumPowerOutput[g])
        Tsu  = value(m.StartupTime[g])
        Tsd  = value(m.ShutdownTime[g])
        Tu0  = value(m.UnitOnT0State[g])
        if (v0==1) and (P0 < Pmin) and (Tu0 > Tsu):
            # Which shut-down interval am I in?
            i = int(round(Tsd - P0/Pmin*Tsd + 1)) 
            t_fixed = m.TimePeriods.first() + Tsd - i
            set_gen_time.add((g, t_fixed))
    return set_gen_time

def thermal_gen_output_limits_startup_lower_rule(m, g, t):
    return (m.UnitOn[g, t] - m.SigmaDn[g, t] - m.SigmaUp[g, t])*m.MinimumPowerOutput[g] + m.SigmaPowerTimesUp[g, t] - m.Slack_startup_lower[g, t] <= m.MinimumPowerAvailable[g, t]

def thermal_gen_output_limits_startup_upper_rule(m, g, t):
    return m.MaximumPowerAvailable[g, t] <= (m.UnitOn[g, t] - m.SigmaUp[g, t])*m.MaximumPowerOutput[g] + m.SigmaPowerTimesUp[g, t] + m.Slack_startup_upper[g, t]

def thermal_gen_output_limits_shutdown_upper_initial_rule(m, g):
    return (m.UnitOnT0[g] - m.SigmaDnT0[g])*m.MinimumPowerOutput[g] + m.SigmaPowerTimesDnT0[g] <= m.PowerGeneratedT0[g]

def thermal_gen_output_limits_shutdown_lower_initial_rule(m, g):
    return (m.UnitOnT0[g] - m.SigmaDnT0[g])*m.MaximumPowerOutput[g] + m.SigmaPowerTimesDnT0[g] >= m.PowerGeneratedT0[g]

def thermal_gen_output_limits_shutdown_lower_rule(m, g, t):
    return (m.UnitOn[g, t] - m.SigmaDn[g, t] - m.SigmaUp[g, t])*m.MinimumPowerOutput[g] + m.SigmaPowerTimesDn[g, t] - m.Slack_shutdown_lower[g, t] <= m.MinimumPowerAvailable[g, t]

def thermal_gen_output_limits_shutdown_upper_rule(m, g, t):
    return m.MaximumPowerAvailable[g, t] <= (m.UnitOn[g, t] - m.SigmaDn[g, t])*m.MaximumPowerOutput[g] + m.SigmaPowerTimesDn[g, t] + m.Slack_shutdown_upper[g, t]

def thermal_gen_output_limits_overlap_startup_rule(m, g, t):
    Tsu  = value(m.StartupTime[g])
    Pmin = value(m.MinimumPowerOutput[g])
    return m.MinimumPowerAvailable[g, t] + m.Slack_overlap_startup[g, t] >= (m.SigmaDn[g, t] + m.SigmaUp[g, t] - 1)*float(Tsu)/Tsu*Pmin

def thermal_gen_output_limits_overlap_shutdown_rule(m, g, t):
    Tsd = value(m.ShutdownTime[g])
    Pmin = value(m.MinimumPowerOutput[g])
    return m.MinimumPowerAvailable[g, t] + m.Slack_overlap_shutdown[g, t] >= (m.SigmaDn[g, t] + m.SigmaUp[g, t] - 1)*float(Tsd-1+1)/Tsd*Pmin

def thermal_gen_rampup_rule(m, g, t):
    if t is m.TimePeriods.first():
        power_last_period = m.PowerGeneratedT0[g]
    else:
        t_prev = m.TimePeriods.prev(t)
        power_last_period = m.PowerGenerated[g, t_prev]
    return m.MaximumPowerAvailable[g, t] - power_last_period - m.Slack_rampup[g, t] <= m.SigmaUp[g,t]*m.MaximumPowerOutput[g] + m.NominalRampUpLimit[g]*(m.UnitOn[g, t] - m.SigmaUp[g, t]) 

def thermal_gen_rampdn_rule(m, g, t):
    if t is m.TimePeriods.first():
        power_last_period = m.PowerGeneratedT0[g]
    else:
        t_prev = m.TimePeriods.prev(t)
        power_last_period = m.PowerGenerated[g, t_prev]
    # return power_last_period - m.PowerGenerated[g, t] - m.Slack_rampdn[g, t] <= (m.SigmaDn[g,t])*m.MaximumPowerOutput[g] + m.NominalRampDownLimit[g]*(m.UnitOn[g, t] - m.SigmaDn[g, t]) 
    return power_last_period - m.PowerGenerated[g, t] - m.Slack_rampdn[g, t] <= (m.SigmaDn[g,t] + m.UnitShutDn[g,t])*m.MinimumPowerOutput[g]/m.ShutdownTime[g] + m.NominalRampDownLimit[g]*(m.UnitOn[g, t] - m.SigmaDn[g, t]) 

def thermal_gen_startup_shutdown_rule(m, g, t):
    if t is m.TimePeriods.first():
        uniton_last_period = m.UnitOnT0[g]
    else:
        uniton_last_period = m.UnitOn[g, m.TimePeriods.prev(t)]
    return m.UnitStartUp[g, t] - m.UnitShutDn[g, t] == m.UnitOn[g, t] - uniton_last_period

def thermal_gen_indicator_startup_rule(m, g, t):
    return m.UnitOn[g, t] >= m.SigmaUp[g, t]

def thermal_gen_indicator_shutdown_rule(m, g, t):
    return m.UnitOn[g, t] >= m.SigmaDn[g, t]

def thermal_gen_indicator_overlap_rule(m, g, t):
    Tsu = value(m.StartupTime[g])
    Tsd = value(m.ShutdownTime[g])
    if t + Tsu + Tsd - 2 <= m.TimePeriods.last():
        return m.UnitStartUp[g, t] + sum(m.UnitShutDn[g, t+i-1] for i in range(1, Tsu+Tsd)) <= 1
    else:
        return Constraint.Skip

def thermal_gen_indicator_shutdown_fixed_rule(m, g, t):
    # print 'shut-down fixed:', g, t
    return m.UnitShutDn[g, t] == 1

# def enforce_generator_output_limits_rule_part_a(m, g, t):
#     return m.MinimumPowerOutput[g]*m.UnitOn[g, t] <= m.PowerGenerated[g,t]

def thermal_gen_output_max_available_rule(m, g, t):
    return m.PowerGenerated[g,t] <= m.MaximumPowerAvailable[g, t]

def thermal_gen_output_min_available_rule(m, g, t):
    return m.MinimumPowerAvailable[g, t] <= m.PowerGenerated[g,t]

# def enforce_generator_output_limits_rule_part_c(m, g, t):
#     return m.MaximumPowerAvailable[g, t] <= m.MaximumPowerOutput[g]*m.UnitOn[g, t]

# Maximum available power of non-thermal units less than forecast
def enforce_renewable_generator_output_limits_rule(m, g, t):
    return  m.MaximumPowerAvailable[g, t] <= m.PowerForecast[g,t]

# Power generation of thermal units by block
def enforce_generator_block_output_rule(m, g, t):
    return m.PowerDollar[g, t] == sum(
           m.BlockPowerGenerated[g,k,t]
           for k in m.Blocks
    )

def enforce_generator_block_output_limit_rule(m, g, k, t):
    return m.BlockPowerGenerated[g,k,t] <= m.BlockSize[g,k]


# impose upper bounds on the maximum power available for each generator in each time period, 
# based on standard and start-up ramp limits.

# the following constraint encodes Constraint 18 defined in Carrion and Arroyo.

# def enforce_max_available_ramp_up_rates_rule(m, g, t):
#     # 4 cases, split by (t-1, t) unit status (RHS is defined as the delta from 
#     # m.PowerGenerated[g, t-1])
#     # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate 
#     #                                    upper bound due to unit being off) 
#     # (0, 1) - unit switching on:  RHS = startup ramp limit 
#     # (1, 0) - unit switching off: RHS = standard ramp limit minus startup ramp 
#     #                                    limit plus maximum power generated (
#     #                                    degenerate upper bound due to unit off)
#     # (1, 1) - unit staying on:    RHS = standard ramp limit
#     if t == m.TimePeriods.first():
#         return m.MaximumPowerAvailable[g, t] - m.SlackRamp1_neg[g, t] <= (
#             m.PowerGeneratedT0[g] 
#             + m.NominalRampUpLimit[g] * m.UnitOnT0[g] 
#             + m.StartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOnT0[g]) 
#             + m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
#         )
#     else:
#         return m.MaximumPowerAvailable[g, t] - m.SlackRamp1_neg[g, t] <= (
#             m.PowerGenerated[g, m.TimePeriods.prev(t)] 
#             + m.NominalRampUpLimit[g] * m.UnitOn[g, m.TimePeriods.prev(t)] 
#             + m.StartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, m.TimePeriods.prev(t)]) 
#             + m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
#         )

# the following constraint encodes Constraint 19 defined in Carrion and Arroyo.

# def enforce_max_available_ramp_down_rates_rule(m, g, t):
#     # 4 cases, split by (t, t+1) unit status
#     # (0, 0) - unit staying off:   RHS = 0 (degenerate upper bound)
#     # (0, 1) - unit switching on:  RHS = maximum generator output minus shutdown 
#     #                                    ramp limit (degenerate upper bound) 
#     #                                    - this is the strangest case.
#     # (1, 0) - unit switching off: RHS = shutdown ramp limit
#     # (1, 1) - unit staying on:    RHS = maximum generator output (degenerate 
#     #                                    upper bound)
#     if t == m.TimePeriods.last():
#         return Constraint.Skip
#     else:
#         return m.MaximumPowerAvailable[g, t] - m.SlackRamp2_neg[g, t] <= (
#             m.MaximumPowerOutput[g] * m.UnitOn[g, m.TimePeriods.next(t)] 
#             + m.ShutdownRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, m.TimePeriods.next(t)])
#         )

# the following constraint encodes Constraint 20 defined in Carrion and Arroyo.

# def enforce_ramp_down_limits_rule(m, g, t):
#     # 4 cases, split by (t-1, t) unit status: 
#     # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate 
#     #                              upper bound)
#     # (0, 1) - unit switching on:  RHS = standard ramp-down limit minus 
#     #                                    shutdown ramp limit plus maximum 
#     #                                    generator output - this is the 
#     #                                    strangest case.
#     # (1, 0) - unit switching off: RHS = shutdown ramp limit 
#     # (1, 1) - unit staying on:    RHS = standard ramp-down limit 
#     if t == m.TimePeriods.first():
#         return m.PowerGeneratedT0[g] - m.PowerGenerated[g, t] - m.SlackRamp3_neg[g, t] <= (
#             m.NominalRampDownLimit[g] * m.UnitOn[g, t] 
#             + m.ShutdownRampLimit[g] * (m.UnitOnT0[g] - m.UnitOn[g, t]) 
#             + m.MaximumPowerOutput[g] * (1 - m.UnitOnT0[g])
#         )
#     else:
#         return (
#             m.PowerGenerated[g, m.TimePeriods.prev(t)] 
#             - m.PowerGenerated[g, t] 
#         ) - m.SlackRamp3_neg[g, t] <= (
#             m.NominalRampDownLimit[g] * m.UnitOn[g, t] 
#             + m.ShutdownRampLimit[g] * (m.UnitOn[g, m.TimePeriods.prev(t)] - m.UnitOn[g, t]) 
#             + m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, m.TimePeriods.prev(t)])
#         )

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
        # for g in m.ThermalGenerators 
        for t in m.TimePeriods
        if m.TimePeriods.value.index(t) < value(m.InitialTimePeriodsOnLine[g])
    ) + m.SlackUpInitial_plus[g] - m.SlackUpInitial_neg[g] == 0.0

# Constraint for each time period after that not involving the initial condition.
def enforce_up_time_constraints_subsequent(m, g, t):
    # Index of time interval starting from 1, plus one since Python index starts 
    # from 0
    i_t = m.TimePeriods.value.index(t) + 1
    if i_t <= value(m.InitialTimePeriodsOnLine[g]):
        # Handled by the EnforceUpTimeConstraintInitial constraint. 
        return Constraint.Skip
    elif i_t <= (value(m.NumTimePeriods) - value(m.MinimumUpTime[g]) + 1): # Maybe only use one value
        # The right-hand side terms below are only positive if the unit was off 
        # in time (t - 1) but on in time t, and the value is the minimum number 
        # of subsequent consecutive time periods that the unit must be on.
        # Note time step in m.TimePeriods must be 1 for this constraint to work. 
        if t is m.TimePeriods.first():
            return sum(
                m.UnitOn[g, n] 
                for n in m.TimePeriods 
                if n >= t and n <= t + value(m.MinimumUpTime[g]) - 1
            ) + m.SlackUpSubsequent_plus[g, t] - m.SlackUpSubsequent_neg[g, t] >= m.MinimumUpTime[g] * (m.UnitOn[g, t] - m.UnitOnT0[g])
        else:
            return sum(
                m.UnitOn[g, n] 
                for n in m.TimePeriods 
                if n >= t and n <= t + value(m.MinimumUpTime[g]) - 1
            ) + m.SlackUpSubsequent_plus[g, t] - m.SlackUpSubsequent_neg[g, t] >= m.MinimumUpTime[g] * (m.UnitOn[g, t] - m.UnitOn[g, m.TimePeriods.prev(t)])
    else:
        # Handle the final (MinimumUpTime[g] - 1) time periods - if a unit is 
        # started up in this interval, it must remain on-line until the end of 
        # the time span.
        if t == m.TimePeriods.first(): # can happen when small time horizons are specified
            return sum(
                m.UnitOn[g, n] - (m.UnitOn[g, t] - m.UnitOnT0[g])
                for n in m.TimePeriods if n >= t
            ) + m.SlackUpSubsequent_plus[g, t] - m.SlackUpSubsequent_neg[g, t] >= 0.0
        else:
            return sum(
                m.UnitOn[g, n] - (m.UnitOn[g, t] - m.UnitOn[g, m.TimePeriods.prev(t)]) 
                for n in m.TimePeriods if n >= t
            ) + m.SlackUpSubsequent_plus[g, t] - m.SlackUpSubsequent_neg[g, t] >= 0.0

#############################################
# Down-time constraints
#############################################

# constraint due to initial conditions.
def enforce_down_time_constraints_initial(m, g):
    if value(m.InitialTimePeriodsOffLine[g]) is 0: 
        return Constraint.Skip
    return sum(
         m.UnitOn[g, t] 
    #    for g in m.ThermalGenerators 
        for t in m.TimePeriods
        if m.TimePeriods.value.index(t) < value(m.InitialTimePeriodsOffLine[g])
   ) + m.SlackDnInitial_plus[g] - m.SlackDnInitial_neg[g] == 0.0


# constraint for each time period after that not involving the initial condition.
def enforce_down_time_constraints_subsequent(m, g, t):
    # Index of time interval starting from 1, plus one since Python index starts
    # from 0
    i_t = m.TimePeriods.value.index(t) + 1
    if i_t <= value(m.InitialTimePeriodsOffLine[g]):
        # handled by the EnforceDownTimeConstraintInitial constraint.
        return Constraint.Skip
    elif i_t <= (value(m.NumTimePeriods) - value(m.MinimumDownTime[g]) + 1):
        # The right-hand side terms below are only positive if the unit was on 
        # in time (t - 1) but on in time, and the value is the minimum number of 
        # subsequent consecutive time periods that the unit must be on.
        if t is m.TimePeriods.first():
            return sum(
                1 - m.UnitOn[g, n]
                for n in m.TimePeriods 
                if n >= t and n <= t + value(m.MinimumDownTime[g]) - 1
            ) + m.SlackDnSubsequent_plus[g, t] - m.SlackDnSubsequent_neg[g, t] >= m.MinimumDownTime[g] * (m.UnitOnT0[g] - m.UnitOn[g, t])
        else:
            return sum(
                1 - m.UnitOn[g, n] 
                for n in m.TimePeriods 
                if n >= t and n <= t + value(m.MinimumDownTime[g]) - 1
            ) + m.SlackDnSubsequent_plus[g, t] - m.SlackDnSubsequent_neg[g, t] >= m.MinimumDownTime[g] * (m.UnitOn[g, m.TimePeriods.prev(t)] - m.UnitOn[g, t])
    else:
        # handle the final (MinimumDownTime[g] - 1) time periods - if a unit is 
        # shut down in this interval, it must remain off-line until the end of 
        # the time span.
        if t is m.TimePeriods.first(): # can happen when small time horizons are specified
            return sum(
                (1 - m.UnitOn[g, n]) - (m.UnitOnT0[g] - m.UnitOn[g, t])
                for n in m.TimePeriods if n >= t
            ) + m.SlackDnSubsequent_plus[g, t] - m.SlackDnSubsequent_neg[g, t] >= 0.0
        else:
            return sum(
                (1 - m.UnitOn[g, n]) - (m.UnitOn[g, m.TimePeriods.prev(t)] - m.UnitOn[g, t]) 
                for n in m.TimePeriods if n >= t
            ) + m.SlackDnSubsequent_plus[g, t] - m.SlackDnSubsequent_neg[g, t] >= 0.0


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
    # return (
    #     m.PowerGenerated[g, t]
    #     - m.RegulatingReserveDnAvailable[g, t]
    #     - m.MinimumPowerOutput[g] * m.UnitOn[g, t]
    # ) >= 0
    return (
        m.PowerGenerated[g, t] 
        - m.RegulatingReserveDnAvailable[g, t] 
        - m.MinimumPowerAvailable[g, t]
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

def powerdollar_rule(m, g, t):
    # The part of generated power that actually counts towards production costs
    return m.PowerDollar[g, t] >= m.PowerGenerated[g, t] - m.BlockSize0[g]

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
    # if t is m.TimePeriods.first():
    #     return m.ShutdownCost[g, t] >= m.ShutdownCostCoefficient[g] * (
    #         m.UnitOnT0[g] - m.UnitOn[g, t]
    #     )
    # else:
    #     return m.ShutdownCost[g, t] >= m.ShutdownCostCoefficient[g] * (
    #         m.UnitOn[g, m.TimePeriods.prev(t)] - m.UnitOn[g, t]
    #     )
    return m.ShutdownCost[g, t] >= m.ShutdownCostCoefficient[g] * m.UnitShutDn[g, t]

def compute_startup_costs_rule(m, g, t):
    # if t is m.TimePeriods.first():
    #     return m.StartupCost[g, t] >= m.StartupCostCoefficient[g] * (
    #         -m.UnitOnT0[g] + m.UnitOn[g, t]
    #     )
    # else:
    #     return m.StartupCost[g, t] >= m.StartupCostCoefficient[g] * (
    #         -m.UnitOn[g, m.TimePeriods.prev(t)] + m.UnitOn[g, t]
    #     )
    return m.StartupCost[g, t] >= m.StartupCostCoefficient[g] * m.UnitStartUp[g, t]

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

# Compute the penalty cost associated with slack variables
def SlackPenalty_rule(m):
#     return 10000000*sum(
#        m.OverCommit[t] + 
#        sum(
#            m.SlackUpSubsequent_plus[g, t] + 
#            m.SlackUpSubsequent_neg[g, t] +
#            m.SlackDnSubsequent_plus[g, t] + 
#            m.SlackDnSubsequent_neg[g, t] +
#            m.SlackRamp1_neg[g, t] + 
#            m.SlackRamp2_neg[g, t] +
#            m.SlackRamp3_neg[g, t]
#            for g in m.ThermalGenerators
#        ) 
#        for t in m.TimePeriods
#    ) + 10000000*sum(
#        m.SlackUpInitial_plus[g] + 
#        m.SlackUpInitial_neg[g] +
#        m.SlackDnInitial_plus[g] + 
#        m.SlackDnInitial_neg[g] 
#        for g in m.ThermalGenerators
#    )
    return 10000000*sum(
       m.OverCommit[t] +
       sum(
            m.Slack_startup_lower[g, t] +
            m.Slack_startup_upper[g, t] +
            m.Slack_shutdown_lower[g, t] +
            m.Slack_shutdown_upper[g, t] +
            m.Slack_overlap_startup[g, t] +
            m.Slack_overlap_shutdown[g, t] +
            m.Slack_rampup[g, t] +
            m.Slack_rampdn[g, t]
           for g in m.ThermalGenerators
       )
       for t in m.TimePeriods
   )

# Objectives
def total_cost_objective_rule(m):
    return (
        m.TotalProductionCost + 
        m.TotalFixedCost + 
        m.TotalCurtailmentCost + 
        m.TotalReserveShortageCost
    ) + m.SlackPenalty

# Additional dispatch limits for RTUC
def EnforceGeneratorOutputLimitsDispacth_rule(m, g, t):
    return m.MaximumPowerAvailable[g, t] <= m.DispatchLimitsUpper[g, t]

def create_model_new(
    network,
    df_busload, # Only bus load, first dimension time starts from 1, no total load
    df_genfor_nonthermal, # Only generation from nonthermal gens, first dim time starts from 1
    ReserveFactor,
    RegulatingReserveFactor,
    nI, # Number of intervals in an hour, typically DAUC: 1, RTUC: 4
    dict_UnitOnT0State=None, # How many time periods the units have been on at T0 from last RTUC model
    dict_PowerGeneratedT0=None, # Initial power generation level at T0 from last RTUC model
    dict_uniton_da=None, # Slow units commitment statuses from DAUC model
    ##############################
    dict_DispatchLimitsUpper=None, # Only apply for slow units
):
    model = ConcreteModel()
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    """SETS & PARAMETERS"""
    ##########################################################
    # The number of time periods under consideration, in addition to the corresponding set.
    model.NumTimePeriods = Param(within=PositiveIntegers, initialize=len(df_busload.index))
    model.TimePeriods    = Set(initialize=df_busload.index, ordered=True) # Time periods must be an ordered set of continuous integers

    # String indentifiers for the sets of different types of generators.
    i_thermal = (network.df_gen['GEN_TYPE']=='Thermal')
    model.AllGenerators        = Set(initialize=network.df_gen.index)
    model.ThermalGenerators    = Set(initialize=network.df_gen[i_thermal].index)
    model.NonThermalGenerators = Set(initialize=network.df_gen[~i_thermal].index)
    model.RenewableGenerators  = Set(initialize=network.df_gen[network.df_gen['GEN_TYPE']=='Renewable'].index)
    model.HydroGenerators      = Set(initialize=network.df_gen[network.df_gen['GEN_TYPE']=='Hydro'].index)
    model.WindGenerators       = Set(initialize=[i for i in network.df_gen.index if i.startswith('wind')])

    # if dict_uniton_da:
    #     set_slow = set( i[0] for i in dict_uniton_da.iterkeys() )
        # model.ThermalGenerators_slow = Set(
        #     initialize=set_slow,
        #     within=model.ThermalGenerators
        # )
        # model.ThermalGenerators_fast = Set(
        #     initialize=model.ThermalGenerators.value.difference(set_slow),
        #     within=model.ThermalGenerators
        # )

    # Set of Generator Blocks Set.
    model.Blocks = Set(initialize = network.df_blockmargcost.columns)
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
    # Number of cost function blockes indexed by (gen, block)
    model.BlockSize = Param(
        model.ThermalGenerators, model.Blocks,
        initialize=network.return_dict_blockoutputlimit()
    )
    model.BlockMarginalCost = Param(
        model.ThermalGenerators, model.Blocks, 
        within=NonNegativeReals,
        initialize=network.return_dict_blockmargcost()
    )
    model.BlockSize0 = Param(
        model.ThermalGenerators,
        initialize=network.df_margcost['Pmax0'].to_dict()
    )
    model.BlockMarginalCost0 = Param(
        model.ThermalGenerators,
        initialize=network.df_margcost['nlcost'].to_dict()
    )

    # String indentifiers for the set of load buses.
    # model.LoadBuses = Set(initialize=load_s_df.columns)
    model.LoadBuses = Set(initialize=df_busload.columns)
    model.Buses     = Set(initialize=network.df_bus.index)
    # Buses indexed by all generators.
    model.GenBuses = Param(model.AllGenerators, initialize=network.df_gen['GEN_BUS'].to_dict())


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

    ##########################################################
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
        model.NonThermalGenerators, 
        model.TimePeriods,
        within=NonNegativeReals,
        initialize=MyDataFrame(df_genfor_nonthermal.T).to_dict_2d(),
        mutable=True
    )

    model.MinimumPowerOutput = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=network.df_gen.loc[i_thermal, 'PMIN'].to_dict()
    )
    model.MaximumPowerOutput = Param(
        model.ThermalGenerators,
        within=NonNegativeReals, 
        initialize=network.df_gen.loc[i_thermal, 'PMAX'].to_dict(),
        validate=maximum_power_output_validator
    )

    # Generator ramp up/down rates. units are MW/interval.
    # Limits for normal time periods
    model.NominalRampUpLimit   = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=(network.df_gen.loc[i_thermal, 'RAMP_10']/nI).to_dict(),
    )
    model.NominalRampDownLimit = Param(
        model.ThermalGenerators,
        within=NonNegativeReals, 
        initialize=(network.df_gen.loc[i_thermal, 'RAMP_10']/nI).to_dict(),
    )

    # Start-up and shut-down time in intervals for thermal gens
    model.StartupTime = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=(network.df_gen.loc[i_thermal, 'STARTUP_TIME']).to_dict(),
        # validate=at_least_generator_minimum_output_validator,
    )
    model.ShutdownTime = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=(network.df_gen.loc[i_thermal, 'SHUTDOWN_TIME']).to_dict(),
        # validate=at_least_generator_minimum_output_validator,
    )

    # Limits for start-up/shut-down
    # model.StartupRampLimit  = Param(
    #     model.ThermalGenerators,
    #     within=NonNegativeReals,
    #     initialize=(network.df_gen.loc[i_thermal, 'STARTUP_RAMP']).to_dict(), # We will use start-up and shut-down time to initialize in the future
    #     validate=at_least_generator_minimum_output_validator
    # )
    # model.ShutdownRampLimit = Param(
    #     model.ThermalGenerators,
    #     within=NonNegativeReals,
    #     initialize=(network.df_gen.loc[i_thermal, 'SHUTDOWN_RAMP']).to_dict(),
    #     validate=at_least_generator_minimum_output_validator
    # )

    # Min number of time periods that a gen must be on-line (off-line) once brought up (down).
    model.MinimumUpTime = Param(
        model.ThermalGenerators,
        within=NonNegativeIntegers,
        initialize=(network.df_gen.loc[i_thermal, 'MINIMUM_UP_TIME']*nI).to_dict(),
        mutable=True
    )
    model.MinimumDownTime = Param(
        model.ThermalGenerators,
        within=NonNegativeIntegers,
        initialize=(network.df_gen.loc[i_thermal, 'MINIMUM_DOWN_TIME']*nI).to_dict(),
        mutable=True
    )

    # if dict_UnitOnT0State:
    #     # Use unit commitment statuses from previous RTUC solutions
    #     model.UnitOnT0State = Param(
    #         model.ThermalGenerators,
    #         within=Integers,
    #         initialize=dict_UnitOnT0State,
    #         validate=t0_state_nonzero_validator,
    #         mutable=True
    #     )
    # else:
    #     model.UnitOnT0State = Param(
    #         model.ThermalGenerators,
    #         within=Integers,
    #         initialize=(network.df_gen.loc[i_thermal, 'GEN_STATUS']*nI).to_dict(),
    #         validate=t0_state_nonzero_validator,
    #         mutable=True
    #     )
    model.UnitOnT0State = Param(
        model.ThermalGenerators,
        within=Integers,
        initialize=(
            dict_UnitOnT0State
            if dict_UnitOnT0State
            else (network.df_gen.loc[i_thermal, 'GEN_STATUS']*nI).to_dict()
        ),
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
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=(
            dict_PowerGeneratedT0 
            if dict_PowerGeneratedT0 
            else network.df_gen.loc[i_thermal, 'PMIN'].to_dict()
        ),
        mutable=True
    )

    # Shutdown and startup cost for each generator, in the literature, these are 
    # often set to 0. However, in order to enforce the start-up and shut-down 
    # profiles, these costs cannot be zero here.
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
        model.ThermalGenerators, 
        model.TimePeriods,
        within=NonNegativeReals, 
        initialize=0.0
    )
    model.RegulatingReserveDnAvailable = Var(
        model.ThermalGenerators, 
        model.TimePeriods,
        within=NonNegativeReals, 
        initialize=0.0
    )
    model.SpinningReserveUpAvailable = Var(
        model.ThermalGenerators, 
        model.TimePeriods, 
        within=NonNegativeReals, 
        initialize=0.0
    )

    # Reserve shortages
    model.RegulatingReserveUpShortage = Var(
        model.TimePeriods,
        within=NonNegativeReals, 
        initialize=0.0
    )
    model.RegulatingReserveDnShortage = Var(
        model.TimePeriods,
        within=NonNegativeReals, 
        initialize=0.0
    )
    model.SpinningReserveUpShortage = Var(
        model.TimePeriods, 
        within=NonNegativeReals, 
        initialize=0.0
    )

    # Generator related variables
    # Indicator variables for each generator, at each time period.
    model.UnitOn = Var(
        model.ThermalGenerators, 
        model.TimePeriods,
        within=Binary, 
        initialize=0
    )
    if dict_uniton_da:
        for k in dict_uniton_da.iterkeys():
            model.UnitOn[k].fix(dict_uniton_da[k])

    model.UnitStartUp = Var(
        model.ThermalGenerators,
        model.TimePeriods,
        bounds=(0, 1),
        within=NonNegativeReals, 
        initialize=0
    )
    model.UnitShutDn = Var(
        model.ThermalGenerators,
        model.TimePeriods,
        bounds=(0, 1),
        within=NonNegativeReals, 
        initialize=0
    )

    # Amount of power produced by each generator, at each time period.
    model.PowerGenerated = Var(
        model.AllGenerators, 
        model.TimePeriods,
        within=NonNegativeReals, 
        initialize=0.0
    )
    # Amount of power that is counted towards production cost, 
    # = PowerGenerated - no load cost
    model.PowerDollar = Var(
        model.ThermalGenerators, 
        model.TimePeriods,
        within=NonNegativeReals, 
        initialize=0.0
    )
    # Amount of power produced by each generator, in each block, at each time period.
    model.BlockPowerGenerated = Var(
        model.ThermalGenerators,
        model.Blocks,
        model.TimePeriods,
        within=NonNegativeReals,
        initialize=0.0
    )

    # Maximum and minimum power output for all generators, at each time period.
    model.MaximumPowerAvailable = Var(
        model.AllGenerators,
        model.TimePeriods,
        within=NonNegativeReals,
        initialize=0.0
    )
    model.MinimumPowerAvailable = Var(
        model.AllGenerators,
        model.TimePeriods,
        within=NonNegativeReals,
        initialize=0.0
    )

    # Costs
    # Production cost associated with each generator, for each time period.
    model.ProductionCost = Var(
        model.ThermalGenerators,
        model.TimePeriods, 
        within=NonNegativeReals,
        initialize=0.0
    )
    # Cost over all generators, for all time periods.
    model.TotalProductionCost = Var(within=NonNegativeReals, initialize=0.0)

    # Startup and shutdown costs for each generator, each time period.
    model.StartupCost = Var(
        model.ThermalGenerators,
        model.TimePeriods, 
        within=NonNegativeReals, 
        initialize=0.0
    )
    model.ShutdownCost = Var(
        model.ThermalGenerators,
        model.TimePeriods, 
        within=NonNegativeReals, 
        initialize=0.0
    )
    model.TotalFixedCost = Var(within=NonNegativeReals, initialize=0.0)

    # Load curtailment penalty cost
    model.BusCurtailment = Var(
        model.LoadBuses,
        model.TimePeriods, 
        within=NonNegativeReals, 
        initialize=0.0
    )
    model.Curtailment = Var(model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    model.TotalCurtailmentCost = Var(initialize=0.0, within=NonNegativeReals)

    # Reserve shortage penalty cost
    model.TotalReserveShortageCost = Var(initialize=0.0, within=NonNegativeReals)

    # Slack variables
    model.OverCommit = Var(model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    model.Slack_startup_lower    = Var(model.ThermalGenerators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    model.Slack_startup_upper    = Var(model.ThermalGenerators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    model.Slack_shutdown_lower   = Var(model.ThermalGenerators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    model.Slack_shutdown_upper   = Var(model.ThermalGenerators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    model.Slack_overlap_startup  = Var(model.ThermalGenerators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    model.Slack_overlap_shutdown = Var(model.ThermalGenerators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    model.Slack_rampup           = Var(model.ThermalGenerators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    model.Slack_rampdn           = Var(model.ThermalGenerators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)

    ############################################
    # Define assisting variable, should we use 
    # actual variable instead of expression?
    ############################################

    model.SigmaUp = Expression(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=sigma_up_rule,
    )
    model.SigmaDn = Expression(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=sigma_dn_rule,
    )
    model.SigmaPowerTimesUp = Expression(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=sigma_power_times_up_rule,
    )
    model.SigmaPowerTimesDn = Expression(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=sigma_power_times_dn_rule,
    )

    model.set_initial_shutdown_power_limits = Set(
        within=model.ThermalGenerators,
        initialize=set_initial_shutdown_power_limits_rule,
    )
    model.SigmaDnT0 = Expression(
        model.set_initial_shutdown_power_limits,
        rule=sigma_dn_initial_rule,
    )
    model.SigmaPowerTimesDnT0 = Expression(
        model.set_initial_shutdown_power_limits,
        rule=sigma_power_times_dn_initial_rule,
    )

    """CONSTRAINTS"""
    ############################################
    # supply-demand constraints                #
    ############################################
    model.SlackPenalty = Expression(
        rule = SlackPenalty_rule
    )
    model.DefineHourlyCurtailment = Constraint(
        model.TimePeriods, rule=definition_hourly_curtailment_rule
    )
    model.ProductionEqualsDemand = Constraint(
        model.TimePeriods, rule=production_equals_demand_rule
    )

    ############################################
    # generation limit constraints #
    ############################################
    model.thermal_gen_output_max_available = Constraint(
        model.AllGenerators,
        model.TimePeriods, 
        rule=thermal_gen_output_max_available_rule,
    )
    model.thermal_gen_output_min_available = Constraint(
        model.AllGenerators,
        model.TimePeriods, 
        rule=thermal_gen_output_min_available_rule,
    )
    model.EnforceRenewableOutputLimits = Constraint(
        model.NonThermalGenerators,
        model.TimePeriods, 
        rule=enforce_renewable_generator_output_limits_rule,
    )

    ############################################
    # Thermal generation start-up/shut-down.
    ############################################

    model.thermal_gen_output_limits_shutdown_upper_initial = Constraint(
        model.set_initial_shutdown_power_limits,
        rule=thermal_gen_output_limits_shutdown_upper_initial_rule,
    )

    model.thermal_gen_output_limits_shutdown_lower_initial = Constraint(
        model.set_initial_shutdown_power_limits,
        rule = thermal_gen_output_limits_shutdown_lower_initial_rule,
    )

    model.thermal_gen_output_limits_startup_lower = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_output_limits_startup_lower_rule,
    )

    model.thermal_gen_output_limits_startup_upper = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_output_limits_startup_upper_rule,
    )

    model.thermal_gen_output_limits_shutdown_lower = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_output_limits_shutdown_lower_rule,
    )

    model.thermal_gen_output_limits_shutdown_upper = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_output_limits_shutdown_upper_rule,
    )

    model.thermal_gen_output_limits_overlap_startup = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_output_limits_overlap_startup_rule,
    )

    model.thermal_gen_output_limits_overlap_shutdown = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_output_limits_overlap_shutdown_rule,
    )

    # Ramp rate constraints
    model.thermal_gen_rampup = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_rampup_rule,
    )

    model.thermal_gen_rampdn = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_rampdn_rule,
    )

    model.thermal_gen_startup_shutdown = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_startup_shutdown_rule,
    )

    # Indicator constraints
    model.thermal_gen_indicator_startup = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_indicator_startup_rule,
    )

    model.thermal_gen_indicator_shutdown = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_indicator_shutdown_rule,
    )

    model.thermal_gen_indicator_overlap = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_indicator_overlap_rule,
    )

    # Sets that the shutdown indicators are fixed
    # model.set_fix_shutdown = Set( dimen=2, initialize=init_set_fix_shutdown )

    # model.thermal_gen_indicator_shutdown_fixed = Constraint(
    #     model.set_fix_shutdown,
    #     rule=thermal_gen_indicator_shutdown_fixed_rule,
    # )

    ############################################
    # generation block outputs constraints #
    ############################################
    model.EnforceGeneratorBlockOutput = Constraint(
        model.ThermalGenerators, 
        model.TimePeriods, 
        rule=enforce_generator_block_output_rule
    )
    model.EnforceGeneratorBlockOutputLimit = Constraint(
        model.ThermalGenerators, 
        model.Blocks, 
        model.TimePeriods, 
        rule=enforce_generator_block_output_limit_rule
    )

    ############################################
    # generation ramping constraints #
    ############################################
    # model.EnforceMaxAvailableRampUpRates = Constraint(
    #     model.ThermalGenerators, model.TimePeriods, 
    #     rule=enforce_max_available_ramp_up_rates_rule
    # )

    # model.EnforceMaxAvailableRampDownRates = Constraint(
    #     model.ThermalGenerators, model.TimePeriods, 
    #     rule=enforce_max_available_ramp_down_rates_rule
    # )

    # model.EnforceNominalRampDownLimits = Constraint(
    #     model.ThermalGenerators, model.TimePeriods, 
    #     rule=enforce_ramp_down_limits_rule
    # )

    #############################################
    # constraints for line capacity limits #
    #############################################
    model.LineFlow = Expression(
        model.EnforcedBranches,
        model.TimePeriods,
        rule=line_flow_rule
    )
    model.EnforceLineCapacityLimitsA = Constraint(
        model.EnforcedBranches, 
        model.TimePeriods, 
        rule=enforce_line_capacity_limits_rule_a
    )   
    model.EnforceLineCapacityLimitsB = Constraint(
        model.EnforcedBranches, 
        model.TimePeriods, 
        rule=enforce_line_capacity_limits_rule_b
    )

    #############################################
    # Minimum online/offline time constriants #
    #############################################
    # model.EnforceUpTimeConstraintsInitial = Constraint(
    #     model.ThermalGenerators, 
    #     rule=enforce_up_time_constraints_initial
    # )

    # model.EnforceUpTimeConstraintsSubsequent = Constraint(
    #     model.ThermalGenerators, model.TimePeriods, 
    #     rule=enforce_up_time_constraints_subsequent
    # )

    # model.EnforceDownTimeConstraintsInitial = Constraint(
    #     model.ThermalGenerators, 
    #     rule=enforce_down_time_constraints_initial
    # )

    # model.EnforceDownTimeConstraintsSubsequent = Constraint(
    #     model.ThermalGenerators, model.TimePeriods, 
    #     rule=enforce_down_time_constraints_subsequent
    # )

    #############################################
    # Available reseves from thermal generators #
    #############################################

    # model.reserve_up_by_maximum_available_power_thermal_constraint = Constraint(
    #     model.ThermalGenerators, 
    #     model.TimePeriods,
    #     rule=reserve_up_by_maximum_available_power_thermal_rule
    # )
    # model.reserve_dn_by_maximum_available_power_thermal_constraint = Constraint(
    #     model.ThermalGenerators, 
    #     model.TimePeriods,
    #     rule=reserve_dn_by_maximum_available_power_thermal_rule
    # )

    # model.reserve_up_by_ramp_thermal_constraint = Constraint(
    #     model.ThermalGenerators, model.TimePeriods,
    #     rule=reserve_up_by_ramp_thermal_rule
    # )
    # model.reserve_dn_by_ramp_thermal_constraint = Constraint(
    #     model.ThermalGenerators, model.TimePeriods,
    #     rule=reserve_dn_by_ramp_thermal_rule
    # )

    #############################################
    # Reserve requirements constraints #
    #############################################
    # model.EnforceSpinningReserveUp = Constraint(
    #     model.TimePeriods, rule=enforce_spinning_reserve_requirement_rule
    # )

    # model.EnforceRegulatingUpReserveRequirements = Constraint(
    #     model.TimePeriods, rule=enforce_regulating_up_reserve_requirement_rule
    # )
    # model.EnforceRegulatingDnReserveRequirements = Constraint(
    #     model.TimePeriods, rule=enforce_regulating_down_reserve_requirement_rule
    # )

    #############################################
    # constraints for computing cost components #
    #############################################

    model.powerdollar_constraint = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=powerdollar_rule,
    )
    
    model.ComputeProductionCost = Constraint(
        model.ThermalGenerators, 
        model.TimePeriods, 
        rule=production_cost_function
    )
    model.ComputeTotalProductionCost = Constraint(rule=compute_total_production_cost_rule)

    model.ComputeStartupCosts = Constraint(
        model.ThermalGenerators, 
        model.TimePeriods, 
        rule=compute_startup_costs_rule
    )
    model.ComputeShutdownCosts = Constraint(
        model.ThermalGenerators, 
        model.TimePeriods, 
        rule=compute_shutdown_costs_rule
    )
    model.ComputeTotalFixedCost = Constraint(rule=compute_total_fixed_cost_rule)
    model.ComputeTotalCurtailmentCost = Constraint(rule=compute_total_curtailment_cost_rule)
    model.ComputeTotalReserveShortageCost = Constraint(rule=compute_total_reserve_shortage_cost_rule)

    model.TotalCostObjective = Objective(
        rule=total_cost_objective_rule, sense=minimize
    )

    #############################################
    # Dispatch limits constriants for slow-ramping units in RTUC
    #############################################
    # if dict_DispatchLimitsUpper:
    #     model.ThermalGenerators_slow = Set(
    #         initialize={k[0] for k in dict_DispatchLimitsUpper.iterkeys()},
    #         within=model.ThermalGenerators,
    #     )
    #     model.DispatchLimitsUpper = Param(
    #         model.ThermalGenerators_slow,
    #         model.TimePeriods,
    #         within=NonNegativeReals, 
    #         initialize=dict_DispatchLimitsUpper,
    #     )
    #     model.EnforceGeneratorOutputLimitsDispacth = Constraint(
    #         model.ThermalGenerators_slow,
    #         model.TimePeriods,
    #         rule=EnforceGeneratorOutputLimitsDispacth_rule
    #     )

    return model

def test_new_model():
    t0 = time()
    content = ''

    nI_da = 1 # Number of DAUC intervals in an hour
    nI_ha = 4 # Number of RTUC intervals in an hour
    nI_ed = 12 # Number of RTED intervals in an hour
    nI_agc = 3600/6 # Number of AGC intervals in an hour

    # Time table, should we add AGC time as well?
    df_timesequence = pd.DataFrame(columns=['tH', 'tQ', 't5'], dtype='int')
    for i in range(1, 25):
        for j in range(1, 5):
            for k in range(1,4):
                index = (i-1)*12+(j-1)*3 + k
                df_timesequence.loc[index, 'tH'] = i
                df_timesequence.loc[index, 'tQ'] = (i-1)*4 + j
                df_timesequence.loc[index, 't5'] = index
    df_timesequence = df_timesequence.astype('int')

    # 118 bus system
    casename = '118'
    csv_bus               = '/home/bxl180002/git/FlexibleRampSCUC/118bus/bus.csv'
    csv_branch            = '/home/bxl180002/git/FlexibleRampSCUC/118bus/branch.csv'
    csv_ptdf              = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ptdf.csv'
    csv_gen               = '/home/bxl180002/git/FlexibleRampSCUC/118bus/generator_data_plexos_withRT.csv'
    csv_marginalcost      = '/home/bxl180002/git/FlexibleRampSCUC/118bus/marginalcost.csv'
    csv_blockmarginalcost = '/home/bxl180002/git/FlexibleRampSCUC/118bus/blockmarginalcost.csv'
    csv_blockoutputlimit  = '/home/bxl180002/git/FlexibleRampSCUC/118bus/blockoutputlimit.csv'
    csv_busload           = '/home/bxl180002/git/FlexibleRampSCUC/118bus/loads.csv'
    csv_genfor            = '/home/bxl180002/git/FlexibleRampSCUC/118bus/generator.csv'
    csv_busload_ha        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ha_loads.csv'
    csv_genfor_ha         = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ha_generator.csv'
    csv_busload_ed        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ed_loads.csv'
    csv_genfor_ed         = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ed_generator.csv'
    csv_busload_agc       = '/home/bxl180002/git/FlexibleRampSCUC/118bus/agc_loads.csv'
    csv_genfor_agc        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/agc_generator.csv'

    # Build network object, will moved to case specific initiation functions, 
    # we need to make sure that the network object contains exactly what we need
    # in the model, no more, no less
    network = Network(csv_bus, csv_branch, csv_ptdf, csv_gen, csv_marginalcost, csv_blockmarginalcost, csv_blockoutputlimit)
    network.df_bus['VOLL'] = 9000
    network.baseMVA = 100

    # This is for the Texas system
    if casename=='TX':
        network.enforce_kv_level(230)

    # This is for 118 bus system
    # # This is to fix a bug in the 118 bus system
    if 'Hydro 31' in network.df_gen.index:
        network.df_gen.drop('Hydro 31', inplace=True)
    # Geo 01 is a thermal gen, set startup and shutdown costs to non-zero to 
    # force UnitStartUp and UnitShutDn being intergers.
    if 'Geo 01' in network.df_gen.index:
        network.df_gen.at['Geo 01', 'STARTUP']  = 50
        network.df_gen.at['Geo 01', 'SHUTDOWN'] = 50
        # network.df_margcost.at['Geo 01', 'nlcost'] = 10
        # network.df_margcost.at['Geo 01', '1']      = 10

    # Add start-up and shut-down time in a quick and dirty way
    network.df_gen.loc[:, 'STARTUP_TIME']  = network.df_gen.loc[:, 'MINIMUM_UP_TIME']
    network.df_gen.loc[:, 'SHUTDOWN_TIME'] = network.df_gen.loc[:, 'MINIMUM_UP_TIME']
    network.df_gen.loc[network.df_gen['STARTUP_TIME']>=12,   'STARTUP_TIME']  = 12
    network.df_gen.loc[network.df_gen['SHUTDOWN_TIME']>=12, 'SHUTDOWN_TIME']  = 12

    # Build network object, will moved to case specific initiation functions
    network.dict_gens = dict()
    network.dict_gens['ALL'] = network.df_gen.index.tolist()
    for t in network.df_gen['GEN_TYPE'].unique():
        network.dict_gens[t] = network.df_gen.loc[network.df_gen['GEN_TYPE']==t, 'GEN_TYPE'].index.tolist()
    network.dict_gens['Non thermal'] = network.df_gen.index.difference(
        network.df_gen.loc[network.df_gen['GEN_TYPE']=='Thermal', 'GEN_TYPE'].index
    ).tolist()
    network.dict_gens['Thermal_slow'] = network.df_gen[
        (network.df_gen['MINIMUM_UP_TIME'] > 1) 
        & 
        (network.df_gen['GEN_TYPE']=='Thermal')
    ].index
    network.dict_gens['Thermal_fast'] = network.df_gen[
        (network.df_gen['MINIMUM_UP_TIME'] <= 1) 
        & 
        (network.df_gen['GEN_TYPE']=='Thermal')
    ].index
    # Bus-generator matrix
    network.ls_bus = network.df_ptdf.columns.tolist()
    network.mat_busgen=np.zeros([len(network.ls_bus), len(network.dict_gens['ALL'])])
    dict_genbus = network.df_gen['GEN_BUS'].to_dict()
    for g, b in dict_genbus.iteritems():
        i = network.ls_bus.index(b)
        j = network.dict_gens['ALL'].index(g)
        network.mat_busgen[i, j] = 1
    df_tmp = pd.DataFrame(0, index=network.ls_bus, columns=network.dict_gens['ALL'])
    for g, b in dict_genbus.iteritems():
        df_tmp.loc[b, g] = 1 # This provides an alternative choice other than network.mat_busgen, should compare their performance later
    # AGC related parameters, need update
    network.df_gen['AGC_MODE'] = 'RAW' # Possible options: NA, RAW, SMOOTH, CPS2
    network.df_gen['DEAD_BAND'] = 5 # MW, 5 MW from FESTIV

    # Prepare day-ahead UC data
    df_busload = pd.read_csv(csv_busload, index_col=0)
    df_genfor  = pd.read_csv(csv_genfor, index_col=0)
    df_busload = MyDataFrame(df_busload.loc[:, df_busload.columns.difference(['LOAD'])])
    df_genfor  = MyDataFrame(df_genfor)
    df_genfor.index = range(1, 25) # Kwami's convention: time starts from 1...
    df_genfor_nonthermal = df_genfor.loc[:, network.df_gen[network.df_gen['GEN_TYPE']!='Thermal'].index]
    df_genfor_nonthermal.fillna(0, inplace=True)

    # Reserve margins, will be move to case specific data
    ReserveFactor = 0.1
    RegulatingReserveFactor = 0.05

    # Test purpose
    ############################################################################
    dict_UnitOnT0State = dict()
    dict_PowerGeneratedT0 = dict()
    for g in network.dict_gens['Thermal']:
        dict_UnitOnT0State[g]    = 12
        dict_PowerGeneratedT0[g] = value(network.df_gen.at[g, 'PMAX'])

    # Start=up test, change the print in the sigma_up rule
    dict_PowerGeneratedT0['CC NG 35'] = 298.29/8*4
    dict_UnitOnT0State['CC NG 35'] = 4

    # Shut-down test, change the print in the fix indicator rule
    dict_PowerGeneratedT0['CC NG 16'] = 503.86/8*3
    dict_UnitOnT0State['CC NG 16'] = 12
    ############################################################################

    ############################################################################
    # Start DAUC
    model = create_model_new(
        network,
        df_busload,
        df_genfor_nonthermal,
        ReserveFactor,
        RegulatingReserveFactor,
        nI=1,
        dict_UnitOnT0State=dict_UnitOnT0State,
        dict_PowerGeneratedT0=dict_PowerGeneratedT0,
    )
    msg = 'Model created at: {:>.2f} s'.format(time() - t0)
    print(msg)
    content += msg
    content += '\n'

    instance = model
    optimizer = SolverFactory('cplex')
    results = optimizer.solve(instance, options={"mipgap":0.001})
    instance.solutions.load_from(results)
    msg = 'Model solved at: {:>.2f} s, objective: {:>.2f}'.format(
            time() - t0,
            value(instance.TotalCostObjective)
        )
    print(msg)
    content += msg
    content += '\n'
    # End of DAUC
    ############################################################################

    set_gens = set()
    for k in instance.UnitStartUp:
        if (value(instance.UnitStartUp[k]) > 0) or (value(instance.UnitShutDn[k]) > 0):
            print k, value(instance.UnitStartUp[k]), value(instance.UnitShutDn[k])
            set_gens.add(k[0])

    df_gen    = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.AllGenerators)
    df_uniton = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_regup  = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_regdn  = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    for g in instance.AllGenerators:
        for t in instance.TimePeriods:
            df_gen.at[t, g] = value(instance.PowerGenerated[g, t])
            if g in instance.ThermalGenerators:
                df_uniton.at[t, g] = value(instance.UnitOn[g, t])
                df_regup.at[t, g]  = value(instance.RegulatingReserveUpAvailable[g, t])
                df_regdn.at[t, g]  = value(instance.RegulatingReserveDnAvailable[g, t])

    ls_dict_therm  = list()
    for g in instance.ThermalGenerators:
        for a in ['PowerGenerated', 'UnitOn', 'UnitStartUp', 'UnitShutDn', 'SigmaUp', 'SigmaDn', 'SigmaPowerTimesUp', 'SigmaPowerTimesDn']:
            attr = getattr(instance, a)
            dict_row = {'Gen': g, 'Var': a}
            for t in instance.TimePeriods:
                dict_row[t] = value(attr[g, t])
            ls_dict_therm.append(dict_row)
    df_therm = pd.DataFrame(ls_dict_therm, columns=['Gen', 'Var']+list(instance.TimePeriods.value))

    print 'Non-zero slack variables:'
    items = ['Slack_startup_lower', 'Slack_startup_upper', 'Slack_shutdown_lower', 'Slack_shutdown_upper', 'Slack_overlap_startup', 'Slack_overlap_shutdown', 'Slack_rampup', 'Slack_rampdn', ]
    for i in items:
        attr = getattr(instance, i)
        for k in attr.iterkeys():
            if value(attr[k]) > 0:
                print i, k, value(attr[k])

    ax1 = plt.subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax1.step(
        df_gen.index,
        df_gen.values.sum(axis=1),
        where='post',
        color='b',
        label='Generation'
    )
    ax1.step(
        df_busload.index,
        df_busload.values.sum(axis=1),
        where='post',
        color='k',
        label='Total load'
    )
    ax1.step(
        df_gen.index,
        df_gen.values.sum(axis=1)-df_regdn.values.sum(axis=1),
        where='post',
        color='b',
        label='REGDN'
    )
    ax1.step(
        df_gen.index,
        df_gen.values.sum(axis=1)+df_regup.values.sum(axis=1),
        where='post',
        color='b',
        label='REGUP'
    )
    ax2.step(
        df_uniton.index,
        df_uniton.values.sum(axis=1),
        where='post',
        color='r',
        label='Committed units'
    )
    plt.legend()
    plt.show()
    IP()

    # for g in set_gens:
    #     plt.step(df_gen.index, df_gen[g], where='post')
    #     plt.step(df_gen.index, [value(instance.MinimumPowerOutput[g])]*24, where='post')
    #     plt.title( g + ', Tsd = {:>g}'.format(value(instance.ShutdownTime[g])))
    #     plt.show()

def test_old_model():
    t0 = time()
    content = ''

    nI_da = 1 # Number of DAUC intervals in an hour
    nI_ha = 4 # Number of RTUC intervals in an hour
    nI_ed = 12 # Number of RTED intervals in an hour
    nI_agc = 3600/6 # Number of AGC intervals in an hour

    # Time table, should we add AGC time as well?
    df_timesequence = pd.DataFrame(columns=['tH', 'tQ', 't5'], dtype='int')
    for i in range(1, 25):
        for j in range(1, 5):
            for k in range(1,4):
                index = (i-1)*12+(j-1)*3 + k
                df_timesequence.loc[index, 'tH'] = i
                df_timesequence.loc[index, 'tQ'] = (i-1)*4 + j
                df_timesequence.loc[index, 't5'] = index
    df_timesequence = df_timesequence.astype('int')

    # 118 bus system
    casename = '118'
    csv_bus               = '/home/bxl180002/git/FlexibleRampSCUC/118bus/bus.csv'
    csv_branch            = '/home/bxl180002/git/FlexibleRampSCUC/118bus/branch.csv'
    csv_ptdf              = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ptdf.csv'
    csv_gen               = '/home/bxl180002/git/FlexibleRampSCUC/118bus/generator_data_plexos_withRT.csv'
    csv_marginalcost      = '/home/bxl180002/git/FlexibleRampSCUC/118bus/marginalcost.csv'
    csv_blockmarginalcost = '/home/bxl180002/git/FlexibleRampSCUC/118bus/blockmarginalcost.csv'
    csv_blockoutputlimit  = '/home/bxl180002/git/FlexibleRampSCUC/118bus/blockoutputlimit.csv'
    csv_busload           = '/home/bxl180002/git/FlexibleRampSCUC/118bus/loads.csv'
    csv_genfor            = '/home/bxl180002/git/FlexibleRampSCUC/118bus/generator.csv'
    csv_busload_ha        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ha_loads.csv'
    csv_genfor_ha         = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ha_generator.csv'
    csv_busload_ed        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ed_loads.csv'
    csv_genfor_ed         = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ed_generator.csv'
    csv_busload_agc       = '/home/bxl180002/git/FlexibleRampSCUC/118bus/agc_loads.csv'
    csv_genfor_agc        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/agc_generator.csv'

    # Build network object, will moved to case specific initiation functions, 
    # we need to make sure that the network object contains exactly what we need
    # in the model, no more, no less
    network = Network(csv_bus, csv_branch, csv_ptdf, csv_gen, csv_marginalcost, csv_blockmarginalcost, csv_blockoutputlimit)
    network.df_bus['VOLL'] = 9000
    network.baseMVA = 100

    # This is for the Texas system
    if casename=='TX':
        network.enforce_kv_level(230)

    # This is for 118 bus system
    # # This is to fix a bug in the 118 bus system
    if 'Hydro 31' in network.df_gen.index:
        network.df_gen.drop('Hydro 31', inplace=True)
    # Geo 01 is a thermal gen, set startup and shutdown costs to non-zero to 
    # force UnitStartUp and UnitShutDn being intergers.
    if 'Geo 01' in network.df_gen.index:
        network.df_gen.at['Geo 01', 'STARTUP']  = 50
        network.df_gen.at['Geo 01', 'SHUTDOWN'] = 50
        # network.df_margcost.at['Geo 01', 'nlcost'] = 10
        # network.df_margcost.at['Geo 01', '1']      = 10

    # Add start-up and shut-down time in a quick and dirty way
    network.df_gen.loc[:, 'STARTUP_TIME']  = network.df_gen.loc[:, 'MINIMUM_UP_TIME']
    network.df_gen.loc[:, 'SHUTDOWN_TIME'] = network.df_gen.loc[:, 'MINIMUM_UP_TIME']
    network.df_gen.loc[network.df_gen['STARTUP_TIME']>=12,   'STARTUP_TIME']  = 12
    network.df_gen.loc[network.df_gen['SHUTDOWN_TIME']>=12, 'SHUTDOWN_TIME']  = 12

    # Build network object, will moved to case specific initiation functions
    network.dict_gens = dict()
    network.dict_gens['ALL'] = network.df_gen.index.tolist()
    for t in network.df_gen['GEN_TYPE'].unique():
        network.dict_gens[t] = network.df_gen.loc[network.df_gen['GEN_TYPE']==t, 'GEN_TYPE'].index.tolist()
    network.dict_gens['Non thermal'] = network.df_gen.index.difference(
        network.df_gen.loc[network.df_gen['GEN_TYPE']=='Thermal', 'GEN_TYPE'].index
    ).tolist()
    network.dict_gens['Thermal_slow'] = network.df_gen[
        (network.df_gen['MINIMUM_UP_TIME'] > 1) 
        & 
        (network.df_gen['GEN_TYPE']=='Thermal')
    ].index
    network.dict_gens['Thermal_fast'] = network.df_gen[
        (network.df_gen['MINIMUM_UP_TIME'] <= 1) 
        & 
        (network.df_gen['GEN_TYPE']=='Thermal')
    ].index
    # Bus-generator matrix
    network.ls_bus = network.df_ptdf.columns.tolist()
    network.mat_busgen=np.zeros([len(network.ls_bus), len(network.dict_gens['ALL'])])
    dict_genbus = network.df_gen['GEN_BUS'].to_dict()
    for g, b in dict_genbus.iteritems():
        i = network.ls_bus.index(b)
        j = network.dict_gens['ALL'].index(g)
        network.mat_busgen[i, j] = 1
    df_tmp = pd.DataFrame(0, index=network.ls_bus, columns=network.dict_gens['ALL'])
    for g, b in dict_genbus.iteritems():
        df_tmp.loc[b, g] = 1 # This provides an alternative choice other than network.mat_busgen, should compare their performance later
    # AGC related parameters, need update
    network.df_gen['AGC_MODE'] = 'RAW' # Possible options: NA, RAW, SMOOTH, CPS2
    network.df_gen['DEAD_BAND'] = 5 # MW, 5 MW from FESTIV

    # Prepare day-ahead UC data
    df_busload = pd.read_csv(csv_busload, index_col=0)
    df_genfor  = pd.read_csv(csv_genfor, index_col=0)
    df_busload = MyDataFrame(df_busload.loc[:, df_busload.columns.difference(['LOAD'])])
    df_genfor  = MyDataFrame(df_genfor)
    df_genfor.index = range(1, 25) # Kwami's convention: time starts from 1...
    df_genfor_nonthermal = df_genfor.loc[:, network.df_gen[network.df_gen['GEN_TYPE']!='Thermal'].index]
    df_genfor_nonthermal.fillna(0, inplace=True)

    # Reserve margins, will be move to case specific data
    ReserveFactor = 0.1
    RegulatingReserveFactor = 0.05

    # Test purpose
    ############################################################################
    dict_UnitOnT0State = dict()
    dict_PowerGeneratedT0 = dict()
    for g in network.dict_gens['Thermal']:
        dict_UnitOnT0State[g]    = 12
        dict_PowerGeneratedT0[g] = value(network.df_gen.at[g, 'PMAX'])

    # Start=up test, change the print in the sigma_up rule
    dict_PowerGeneratedT0['CC NG 35'] = 298.29/8*4
    dict_UnitOnT0State['CC NG 35'] = 4

    # Shut-down test, change the print in the fix indicator rule
    dict_PowerGeneratedT0['CC NG 16'] = 503.86/8*3
    dict_UnitOnT0State['CC NG 16'] = 12
    ############################################################################

    ############################################################################
    # Start DAUC, modify that first
    model = create_model(
        network,
        df_busload,
        df_genfor_nonthermal,
        ReserveFactor,
        RegulatingReserveFactor,
        nI=1,
        dict_UnitOnT0State=dict_UnitOnT0State,
        dict_PowerGeneratedT0=dict_PowerGeneratedT0,
    )

    i_thermal = (network.df_gen['GEN_TYPE']=='Thermal')
    model.StartupTime = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=(network.df_gen.loc[i_thermal, 'STARTUP_TIME']).to_dict(),
        # validate=at_least_generator_minimum_output_validator,
    )
    model.ShutdownTime = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=(network.df_gen.loc[i_thermal, 'SHUTDOWN_TIME']).to_dict(),
        # validate=at_least_generator_minimum_output_validator,
    )

    model.UnitStartUp = Var(
        model.ThermalGenerators,
        model.TimePeriods,
        bounds=(0, 1),
        within=NonNegativeReals, 
        initialize=0
    )
    model.UnitShutDn = Var(
        model.ThermalGenerators,
        model.TimePeriods,
        bounds=(0, 1),
        within=NonNegativeReals, 
        initialize=0
    )

    model.thermal_gen_startup_shutdown = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=thermal_gen_startup_shutdown_rule,
    )

    model.SigmaUp = Expression(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=sigma_up_rule,
    )
    model.SigmaDn = Expression(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=sigma_dn_rule,
    )
    model.SigmaPowerTimesUp = Expression(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=sigma_power_times_up_rule,
    )
    model.SigmaPowerTimesDn = Expression(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=sigma_power_times_dn_rule,
    )

    model.del_component(model.ComputeStartupCosts)
    model.del_component(model.ComputeStartupCosts_index)
    model.del_component(model.ComputeShutdownCosts)
    model.del_component(model.ComputeShutdownCosts_index)
    model.ComputeStartupCosts = Constraint(
        model.ThermalGenerators, 
        model.TimePeriods, 
        rule=compute_startup_costs_rule
    )
    model.ComputeShutdownCosts = Constraint(
        model.ThermalGenerators, 
        model.TimePeriods, 
        rule=compute_shutdown_costs_rule
    )

    msg = 'Model created at: {:>.2f} s'.format(time() - t0)
    print(msg)
    content += msg
    content += '\n'

    instance = model
    optimizer = SolverFactory('cplex')
    results = optimizer.solve(instance, options={"mipgap":0.001})
    instance.solutions.load_from(results)
    msg = 'Model solved at: {:>.2f} s, objective: {:>.2f}'.format(
            time() - t0,
            value(instance.TotalCostObjective)
        )
    print(msg)
    content += msg
    content += '\n'
    # End of DAUC
    ############################################################################

    set_gens = set()
    for k in instance.UnitStartUp:
        if (value(instance.UnitStartUp[k]) > 0) or (value(instance.UnitShutDn[k]) > 0):
            print k, value(instance.UnitStartUp[k]), value(instance.UnitShutDn[k])
            set_gens.add(k[0])

    df_gen    = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.AllGenerators)
    df_uniton = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_regup  = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_regdn  = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    for g in instance.AllGenerators:
        for t in instance.TimePeriods:
            df_gen.at[t, g] = value(instance.PowerGenerated[g, t])
            if g in instance.ThermalGenerators:
                df_uniton.at[t, g] = value(instance.UnitOn[g, t])
                df_regup.at[t, g]  = value(instance.RegulatingReserveUpAvailable[g, t])
                df_regdn.at[t, g]  = value(instance.RegulatingReserveDnAvailable[g, t])

    ls_dict_therm  = list()
    for g in instance.ThermalGenerators:
        for a in ['PowerGenerated', 'UnitOn', 'UnitStartUp', 'UnitShutDn', 'SigmaUp', 'SigmaDn', 'SigmaPowerTimesUp', 'SigmaPowerTimesDn']:
            attr = getattr(instance, a)
            dict_row = {'Gen': g, 'Var': a}
            for t in instance.TimePeriods:
                dict_row[t] = value(attr[g, t])
            ls_dict_therm.append(dict_row)
    df_therm = pd.DataFrame(ls_dict_therm, columns=['Gen', 'Var']+list(instance.TimePeriods.value))

    ax1 = plt.subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax1.step(
        df_gen.index,
        df_gen.values.sum(axis=1),
        where='post',
        color='b',
        label='Generation'
    )
    ax1.step(
        df_busload.index,
        df_busload.values.sum(axis=1),
        where='post',
        color='k',
        label='Total load'
    )
    ax1.step(
        df_gen.index,
        df_gen.values.sum(axis=1)-df_regdn.values.sum(axis=1),
        where='post',
        color='b',
        label='REGDN'
    )
    ax1.step(
        df_gen.index,
        df_gen.values.sum(axis=1)+df_regup.values.sum(axis=1),
        where='post',
        color='b',
        label='REGUP'
    )
    ax2.step(
        df_uniton.index,
        df_uniton.values.sum(axis=1),
        where='post',
        color='r',
        label='Committed units'
    )
    plt.legend()
    plt.show()
    IP()

if __name__ == "__main__":
    test_new_model()
    # test_old_model()