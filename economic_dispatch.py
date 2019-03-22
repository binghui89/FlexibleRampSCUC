from pyomo.environ import *
from SCUC_RampConstraint_3 import MyDataFrame

def dispatch_limits(instance_ha, ls_t_dispatch, df_timesequence):
    dict_upperdispatchlimit = dict()
    dict_lowerdispatchlimit = dict()
    for g in instance_ha.ThermalGenerators.iterkeys():
        for t5 in ls_t_dispatch:
            tQ = df_timesequence.loc[df_timesequence['t5'] == t5, 'tQ'].tolist()[0] # There should be only one value in the list
            dict_upperdispatchlimit[g, t5] = max(
                0,
                value(
                    instance_ha.PowerGenerated[g, tQ] + 
                    instance_ha.RegulatingReserveUpAvailable[g, tQ] +
                    instance_ha.SpinningReserveUpAvailable[g, tQ]
                )
            )
            dict_lowerdispatchlimit[g, t5] = max(
                0,
                value(
                    instance_ha.PowerGenerated[g, tQ] - 
                    instance_ha.RegulatingReserveDnAvailable[g, tQ]
                )
            )
    return dict_upperdispatchlimit, dict_lowerdispatchlimit

def build_sced_model(
    #########
    # The following parameters are the same as in the DA/RTUC model
    network,
    df_busload, # Only bus load, first dimension time starts from 1, no total load. For ED model, there should be only 1 row
    df_genfor_nonthermal, # Only generation from nonthermal gens, first dim time starts from 1
    ReserveFactor, # 
    RegulatingReserveFactor, # 
    nI, # Number of intervals in an hour, typically DAUC: 1, RTUC: 4
    dict_UnitOnT0State=None, #
    dict_PowerGeneratedT0=None, # Initial power generation level at T0 from last RTUC model
    dict_uniton_da=None, # Commitment statuses of ALL units from RTUC model
    #########
    # The following parameters only apply for the ED model
    # dict_UpperDispatchLimit=None,
    # dict_LowerDispatchLimit=None,
):

    model = ConcreteModel()
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    """SETS & PARAMETERS"""
    model.NumTimePeriods = Param(within=PositiveIntegers, initialize=len(df_busload.index))
    model.TimePeriods    = Set(initialize=df_busload.index, ordered=True)

    ##########################################################
    # string indentifiers for the sets of different types of generators. #
    ##########################################################
    i_thermal = (network.df_gen['GEN_TYPE']=='Thermal')
    model.AllGenerators        = Set(initialize=network.df_gen.index)
    model.ThermalGenerators    = Set(initialize=network.df_gen[i_thermal].index)
    model.NonThermalGenerators = Set(initialize=network.df_gen[~i_thermal].index)
    model.RenewableGenerators  = Set(initialize=network.df_gen[network.df_gen['GEN_TYPE']=='Renewable'].index)
    model.HydroGenerators      = Set(initialize=network.df_gen[network.df_gen['GEN_TYPE']=='Hydro'].index)
    model.WindGenerators       = Set(initialize=[i for i in network.df_gen.index if i.startswith('wind')])
    # model.NonFlexGen           = Set(initialize=gen_df[gen_df['FLEX_TYPE']=='NonFlexible'].index) # This set seems to be never used...

    ##########################################################
    # Set of Generator Blocks Set.                               #
    ##########################################################
    model.Blocks = Set(initialize = network.df_blockmargcost.columns)
    #model.GenNumBlocks = Param(model.ThermalGenerators, initialize=margcost_df['nblock'].to_dict())
    model.BlockSize = Param(
        model.ThermalGenerators, 
        model.Blocks, 
        initialize=network.return_dict_blockoutputlimit()
    )
    model.BlockMarginalCost = Param(
        model.ThermalGenerators, 
        model.Blocks, 
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

    ##########################################################
    # string indentifiers for the set of load buses. #
    ##########################################################

    model.LoadBuses = Set(initialize=df_busload.columns)
    model.Buses     = Set(initialize=network.df_bus.index)

    ##########################################################
    # string indentifiers for the set of thermal generators buses. #
    ##########################################################

    model.GenBuses = Param(model.AllGenerators, initialize=network.df_gen['GEN_BUS'].to_dict())

    ##########################################################
    # string indentifiers for the set of branches. #
    ##########################################################

    model.Branches         = Set(initialize=network.df_branch.index)
    model.EnforcedBranches = Set(initialize=network.set_valid_branch)

    #################################################################
    # Line capacity limits: units are MW. #
    #################################################################

    model.LineLimits = Param(
        model.Branches,
        # model.EnforcedBranches,
        within=NonNegativeReals,
        initialize = network.df_branch['RATE_A'].to_dict()
    )

    #################################################################
    # PTDF. #
    #################################################################

    #model.PTDF = Param(model.Buses, model.Branches, within=Reals, initialize=ptdf_dict)
    model.PTDF = Param(
        model.Branches, model.Buses,
        within=Reals,
        initialize=network.return_dict_ptdf(tol=None),
        default=0.0
    )

    #################################################################
    # the global system demand, for each time period. units are MW. #
    #################################################################
    model.Demand = Param(
        model.TimePeriods,
        within=NonNegativeReals, 
        initialize=df_busload.sum(axis=1).to_dict(),
        mutable=True
    ) # Total system demand
    model.BusDemand = Param(
        model.LoadBuses,
        model.TimePeriods,
        within=NonNegativeReals, 
        initialize=MyDataFrame(df_busload.T).to_dict_2d(),
        mutable=True
    ) # Bus specific demand

    model.BusVOLL = Param(
        model.LoadBuses,
        within=NonNegativeReals,
        initialize=network.df_bus[ network.df_bus['PD']>0 ][ 'VOLL' ].to_dict()
    )

    # Power forecasts

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
   
    def maximum_power_output_validator(m, v, g):
        return v >= value(m.MinimumPowerOutput[g])

    model.MaximumPowerOutput = Param(
        model.ThermalGenerators,
        within=NonNegativeReals, 
        initialize=network.df_gen.loc[i_thermal, 'PMAX'].to_dict(),
        validate=maximum_power_output_validator
    )

    #################################################
    # generator ramp up/down rates. units are MW/interval. #
    #################################################

    # # limits for normal time periods
    # model.UpperDispatchLimit = Param(
    #     model.ThermalGenerators,
    #     model.TimePeriods,
    #     within=NonNegativeReals,
    #     initialize=dict_UpperDispatchLimit,
    #     mutable=True
    # )
    # model.LowerDispatchLimit = Param(
    #     model.ThermalGenerators,
    #     model.TimePeriods,
    #     within=NonNegativeReals,
    #     initialize=dict_LowerDispatchLimit,
    #     mutable=True
    # )
    #model.NominalRampUpLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=genth_df['RAMP_10'].to_dict())
    #model.NominalRampDownLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=genth_df['RAMP_10'].to_dict())

    # model.MaximumRamp = Param(
    #     model.ThermalGenerators,
    #     within=NonNegativeReals,
    #     initialize=(network.df_gen.loc[i_thermal, 'RAMP_10']).to_dict()
    # ) # Note this is ramp rate per interval
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

    # Limits for start-up/shut-down
    # Limits for time periods in which generators are brought on or off-line. 
    # Must not be less than the generator minimum output. 
    def at_least_generator_minimum_output_validator(m, v, g):
        return v >= m.MinimumPowerOutput[g]
    model.StartupRampLimit  = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=(network.df_gen.loc[i_thermal, 'STARTUP_RAMP']).to_dict(),
        validate=at_least_generator_minimum_output_validator
    )
    model.ShutdownRampLimit = Param(
        model.ThermalGenerators,
        within=NonNegativeReals,
        initialize=(network.df_gen.loc[i_thermal, 'SHUTDOWN_RAMP']).to_dict(),
        validate=at_least_generator_minimum_output_validator
    )

    # indicator variables for each generator, at each time period.
    model.UnitOn = Param(
        model.ThermalGenerators,
        model.TimePeriods,
        within=Binary,
        initialize=(
            dict_uniton_da
            if dict_uniton_da
            else 0
        ),
        mutable=True
    )

    def t0_state_nonzero_validator(m, v, g):
        return v != 0
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

    def t0_unit_on_rule(m, g):
        return int( value(m.UnitOnT0State[g]) >= 1 )
    model.UnitOnT0 = Param(
        model.ThermalGenerators,
        within=Binary,
        initialize=t0_unit_on_rule,
        mutable=True
    )

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

    model.ReserveFactor = Param(
        within=Reals, initialize=ReserveFactor, default=0.0, mutable=True
    )
    model.RegulatingReserveFactor = Param(
        within=Reals, initialize=RegulatingReserveFactor, default=0.0, mutable=True
    )
    # Spinning and Regulating Reserves requirements
    def _reserve_requirement_rule(m, t):
        return m.ReserveFactor*sum(value(m.BusDemand[b,t]) for b in m.LoadBuses)

    def _regulating_requirement_rule(m, t):
        return m.RegulatingReserveFactor*sum(value(m.BusDemand[b,t]) for b in m.LoadBuses)

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
    #==============================================================================
    #  VARIABLE DEFINITION
    #==============================================================================
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

    # amount of power produced by each generator, at each time period.
    model.PowerGenerated = Var(
        model.AllGenerators,
        model.TimePeriods,
        within=NonNegativeReals,
        initialize=0.0
    )
    # amount of power produced by each generator, in each block, at each time period.
    model.BlockPowerGenerated = Var(
        model.ThermalGenerators,
        model.Blocks,
        model.TimePeriods,
        within=NonNegativeReals,
        initialize=0.0
    )
    
    # Maximum power output for each generator, at each time period.
    model.MaximumPowerAvailable = Var(
        model.AllGenerators,
        model.TimePeriods,
        within=NonNegativeReals,
        initialize=0.0
    )

    ###################
    # cost components #
    ###################
   
    # production cost associated with each generator, for each time period.
    model.ProductionCost = Var(
        model.ThermalGenerators,
        model.TimePeriods,
        within=NonNegativeReals,
        initialize=0.0
    )
    # cost over all generators, for all time periods.
    model.TotalProductionCost = Var(within=NonNegativeReals, initialize=0.0)

    # Load curtailment penalty cost
    model.BusCurtailment = Var(
        model.LoadBuses,
        model.TimePeriods,
        initialize=0.0,
        within=NonNegativeReals
    )
    model.Curtailment = Var(model.TimePeriods,initialize=0.0, within=NonNegativeReals)
    model.TotalCurtailmentCost = Var(initialize=0.0, within=NonNegativeReals)

    # Reserve shortage penalty cost
    model.TotalReserveShortageCost = Var(initialize=0.0, within=NonNegativeReals)
    
    """CONSTRAINTS"""
    ############################################
    # supply-demand constraints                #
    ############################################
    # meet the demand at each time period.
    # encodes Constraint 2 in Carrion and Arroyo.

    def enforce_bus_curtailment_limits_rule(m, b, t):
        return m.BusCurtailment[b, t]<= m.BusDemand[b, t]
    model.EnforceBusCurtailmentLimits = Constraint(
        model.LoadBuses,
        model.TimePeriods,
        rule=enforce_bus_curtailment_limits_rule
    )

    def definition_hourly_curtailment_rule(m, t):
        return m.Curtailment[t] == sum(m.BusCurtailment[b, t] for b in m.LoadBuses)
    model.DefineHourlyCurtailment = Constraint(
        model.TimePeriods, 
        rule=definition_hourly_curtailment_rule
    ) 

    def production_equals_demand_rule(m, t):
        return sum(m.PowerGenerated[g, t] for g in m.AllGenerators) + m.Curtailment[t] == m.Demand[t]

    # def production_equals_demand_rule_b(m):
    #     return sum(m.PowerGenerated[g] for g in m.AllGenerators)  >= m.Demand

    model.ProductionEqualsDemand = Constraint(
        model.TimePeriods, 
        rule=production_equals_demand_rule
    )

    ############################################
    # generation limit constraints #
    ############################################
    # enforce the generator power output limits on a per-period basis.
    # the maximum power available at any given time period is dynamic,
    # bounded from above by the maximum generator output.
    
    # the following three constraints encode Constraints 16 and 17 defined in Carrion and Arroyo.
    
    # NOTE: The expression below is what we really want - however, due to a pyomo bug, we have to split it into two constraints:
    # m.MinimumPowerOutput[g] * m.UnitOn[g, t] <= m.PowerGenerated[g,t] <= m.MaximumPowerAvailable[g, t]
    # When fixed, merge back parts "a" and "b", leaving two constraints.

    # def enforce_generator_output_limits_rule_part_c(m, g, t):
    #     return m.PowerGenerated[g, t] >= m.LowerDispatchLimit[g, t]
    # def enforce_generator_output_limits_rule_part_d(m, g, t):
    #     return m.PowerGenerated[g, t] <= m.UpperDispatchLimit[g, t]
    # model.EnforceGeneratorOutputLimitsPartC = Constraint(
    #     model.ThermalGenerators,
    #     model.TimePeriods,
    #     rule=enforce_generator_output_limits_rule_part_c
    # )
    # model.EnforceGeneratorOutputLimitsPartD = Constraint(
    #     model.ThermalGenerators,
    #     model.TimePeriods,
    #     rule=enforce_generator_output_limits_rule_part_d
    # )
    def enforce_generator_output_limits_rule_part_a(m, g, t):
        return m.MinimumPowerOutput[g]*m.UnitOn[g, t] <= m.PowerGenerated[g,t]
    def enforce_generator_output_limits_rule_part_b(m, g, t):
        return m.PowerGenerated[g,t] <= m.MaximumPowerAvailable[g, t]
    def enforce_generator_output_limits_rule_part_c(m, g, t):
        return m.MaximumPowerAvailable[g, t] <= m.MaximumPowerOutput[g]*m.UnitOn[g, t]
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

    # def enforce_renewable_generator_output_limits_rule(m, g, t):
    #     return  m.PowerGenerated[g, t]<= m.PowerForecast[g, t]
    # model.EnforceRenewableOutputLimits = Constraint(
    #     model.NonThermalGenerators,
    #     model.TimePeriods,
    #     rule=enforce_renewable_generator_output_limits_rule
    # )
    def enforce_renewable_generator_output_limits_rule(m, g, t):
        return  m.MaximumPowerAvailable[g, t]<= m.PowerForecast[g,t]
    model.EnforceRenewableOutputLimits = Constraint(
        model.NonThermalGenerators, model.TimePeriods, 
        rule=enforce_renewable_generator_output_limits_rule
    )


    ############################################
    # generation ramping constraints #
    ############################################
    # def enforce_generator_ramp_limits_rule_part_a(m, g, t):
    #     # if value(m.Start)==1 and value(m.Slot)==1:
    #     if t == m.TimePeriods.first():
    #         return m.PowerGenerated[g, t] - m.PowerGeneratedT0[g] <= m.MaximumRamp[g]
    #     else:
    #         t_prev = m.TimePeriods.prev(t)
    #         return m.PowerGenerated[g, t] - m.PowerGenerated[g, t_prev] <= m.MaximumRamp[g]
    # def enforce_generator_ramp_limits_rule_part_b(m, g, t):
    #     if t == m.TimePeriods.first():
    #         return -m.PowerGenerated[g, t] + m.PowerGeneratedT0[g] <= m.MaximumRamp[g]
    #     else:
    #         t_prev = m.TimePeriods.prev(t)
    #         return -m.PowerGenerated[g, t] + m.PowerGeneratedT0[g, t_prev] <= m.MaximumRamp[g]
    # model.EnforceGeneratorRampLimitsPartA = Constraint(
    #     model.ThermalGenerators,
    #     model.TimePeriods,
    #     rule=enforce_generator_ramp_limits_rule_part_a
    # )
    # model.EnforceGeneratorRampLimitsPartB = Constraint(
    #     model.ThermalGenerators,
    #     model.TimePeriods,
    #     rule=enforce_generator_ramp_limits_rule_part_b
    # )
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
        if t == m.TimePeriods.first():
            return m.MaximumPowerAvailable[g, t] <= (
                m.PowerGeneratedT0[g] 
                + m.NominalRampUpLimit[g] * m.UnitOnT0[g] 
                + m.StartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOnT0[g]) 
                + m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
            )
        else:
            return m.MaximumPowerAvailable[g, t] <= (
                m.PowerGenerated[g, m.TimePeriods.prev(t)] 
                + m.NominalRampUpLimit[g] * m.UnitOn[g, m.TimePeriods.prev(t)] 
                + m.StartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, m.TimePeriods.prev(t)]) 
                + m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
            )
    def enforce_max_available_ramp_down_rates_rule(m, g, t):
        # 4 cases, split by (t, t+1) unit status
        # (0, 0) - unit staying off:   RHS = 0 (degenerate upper bound)
        # (0, 1) - unit switching on:  RHS = maximum generator output minus shutdown 
        #                                    ramp limit (degenerate upper bound) 
        #                                    - this is the strangest case.
        # (1, 0) - unit switching off: RHS = shutdown ramp limit
        # (1, 1) - unit staying on:    RHS = maximum generator output (degenerate 
        #                                    upper bound)
        if t == m.TimePeriods.last():
            return Constraint.Skip
        else:
            return m.MaximumPowerAvailable[g, t] <= (
                m.MaximumPowerOutput[g] * m.UnitOn[g, m.TimePeriods.next(t)] 
                + m.ShutdownRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, m.TimePeriods.next(t)])
            )
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
        if t == m.TimePeriods.first():
            return m.PowerGeneratedT0[g] - m.PowerGenerated[g, t] <= (
                m.NominalRampDownLimit[g] * m.UnitOn[g, t] 
                + m.ShutdownRampLimit[g] * (m.UnitOnT0[g] - m.UnitOn[g, t]) 
                + m.MaximumPowerOutput[g] * (1 - m.UnitOnT0[g])
            )
        else:
            return (
                m.PowerGenerated[g, m.TimePeriods.prev(t)] 
                - m.PowerGenerated[g, t] 
            ) <= (
                m.NominalRampDownLimit[g] * m.UnitOn[g, t] 
                + m.ShutdownRampLimit[g] * (m.UnitOn[g, m.TimePeriods.prev(t)] - m.UnitOn[g, t]) 
                + m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, m.TimePeriods.prev(t)])
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

    ############################################
    # generation block outputs constraints #
    ############################################

    def enforce_generator_block_output_rule(m, g, t):
        # return m.PowerGenerated[g] == sum(m.BlockPowerGenerated[g,k] for k in m.Blocks) + m.UnitOn[g]*margcost_df.loc[g,'Pmax0']
        return m.PowerGenerated[g, t] == (
        m.UnitOn[g, t]*m.BlockSize0[g] + 
        sum(
           m.BlockPowerGenerated[g,k,t]
           for k in m.Blocks
       )
    )
    def enforce_generator_block_output_limit_rule(m, g, k, t):
        return m.BlockPowerGenerated[g, k, t] <= m.BlockSize[g, k]
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

    #############################################
    # constraints for line capacity limits #
    #############################################

    # def enforce_line_capacity_limits_rule_a(m, l):
    #     return sum(ptdf_dict[l][m.GenBuses[g]]*m.PowerGenerated[g] for g in m.AllGenerators) - sum(ptdf_dict[l][b]*(m.BusDemand[b] - m.BusCurtailment[b]) for b in m.LoadBuses) <= m.LineLimits[l]
    # def enforce_line_capacity_limits_rule_b(m, l):
    #     return sum(ptdf_dict[l][m.GenBuses[g]]*m.PowerGenerated[g] for g in m.AllGenerators) - sum(ptdf_dict[l][b]*(m.BusDemand[b] - m.BusCurtailment[b]) for b in m.LoadBuses) >= -m.LineLimits[l]

    def line_flow_rule(m, l, t):
        # This is an expression of the power flow on bus b in time t, defined here
        # to save time.
        return sum(
            # ptdf_dict[l][m.GenBuses[g]]*m.PowerGenerated[g,t] 
            m.PTDF[l, m.GenBuses[g]]*m.PowerGenerated[g, t] 
            for g in m.AllGenerators
        ) - sum(
            # ptdf_dict[l][b]*(m.BusDemand[b,t] - m.BusCurtailment[b,t]) 
            m.PTDF[l, b]*(m.BusDemand[b, t] - m.BusCurtailment[b, t]) 
            for b in m.LoadBuses
        )
    def enforce_line_capacity_limits_rule_a(m, l, t):
        return m.LineFlow[l, t] <= m.LineLimits[l]
    def enforce_line_capacity_limits_rule_b(m, l, t):
        return m.LineFlow[l, t] >= -m.LineLimits[l]
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
    # Available reseves from thermal generators #
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

    #############################################
    # Reserve requirements constraints #
    #############################################
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

    model.EnforceSpinningReserveUp = Constraint(
        model.TimePeriods, rule=enforce_spinning_reserve_requirement_rule
    )

    model.EnforceRegulatingUpReserveRequirements = Constraint(
        model.TimePeriods, rule=enforce_regulating_up_reserve_requirement_rule
    )
    model.EnforceRegulatingDnReserveRequirements = Constraint(
        model.TimePeriods, rule=enforce_regulating_down_reserve_requirement_rule
    )

    #############################################
    # constraints for computing cost components #
    #############################################

    def production_cost_function(m, g, t):
        #  return m.ProductionCost[g] == sum(value(m.BlockMarginalCost[g,k])*(m.BlockPowerGenerated[g,k]) for k in m.Blocks) + m.UnitOn[g]*margcost_df.loc[g,'nlcost']
        return m.ProductionCost[g, t] == (
        m.UnitOn[g, t]*m.BlockMarginalCost0[g]
        + sum(
            value(m.BlockMarginalCost[g, k])*(m.BlockPowerGenerated[g, k, t])
            for k in m.Blocks
        )
    )
    model.ComputeProductionCost = Constraint(
        model.ThermalGenerators,
        model.TimePeriods,
        rule=production_cost_function
    )
    #---------------------------------------
    
    # compute the per-generator, per-time period production costs. this is a "simple" piecewise linear construct.
    # the first argument to piecewise is the index set. the second and third arguments are respectively the input and output variables. 
    """
    model.ComputeProductionCosts = Piecewise(model.ThermalGenerators * model.TimePeriods, model.ProductionCost, model.PowerGenerated, pw_pts=model.PowerGenerationPiecewisePoints, f_rule=production_cost_function, pw_constr_type='LB')
    """
    # compute the total production costs, across all generators and time periods.
    def compute_total_production_cost_rule(m):
        return m.TotalProductionCost == sum(
            m.ProductionCost[g, t]
            for g in m.ThermalGenerators 
            for t in m.TimePeriods
        )
    model.ComputeTotalProductionCost = Constraint(rule=compute_total_production_cost_rule)


    def compute_total_curtailment_cost_rule(m):
        return m.TotalCurtailmentCost == sum(
            m.BusVOLL[b]* m.BusCurtailment[b, t]
            for b in m.LoadBuses
            for t in m.TimePeriods
        )
    model.ComputeTotalCurtailmentCost = Constraint(rule=compute_total_curtailment_cost_rule)

    #---------------------------------------------------------------

    #-------------------------------------------------------------
    # Objectives
    #
   
    def total_cost_objective_rule(m):
        return m.TotalProductionCost + m.TotalCurtailmentCost
    model.TotalCostObjective = Objective(rule=total_cost_objective_rule, sense=minimize)

    return model
