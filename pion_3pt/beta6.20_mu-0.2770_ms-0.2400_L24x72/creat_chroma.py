import sys
class creat_chroma_ini:
    def __init__(
        self,
        lattice_size: list[int],
        conf_id: int,
        hadron:str = None,
        conf_dir:str = None,
        out_path:str = None,
        gauge:str = 'default_gauge_field',
        
    ) -> None:
        self.Nx,self.Ny,self.Nz,self.Nt = lattice_size
        self.lattice_size = lattice_size
        self.conf_dir = conf_dir
        self.out_path = out_path
        self.conf_id = conf_id
        self.hadron = hadron
        self.gauge = gauge
    
    def begin(
        self,
        name:str=''
        ):
        print(f'''<?xml version="1.0"?>
    <chroma>
        <annotation>
            Quark project 
                {name}
        </annotation>
        <Param> 
            <InlineMeasurements>  ''')
        
    def ERASE_NAMED_OBJECT(
        self,
        name
        ):
        print(f'''
            <elem>
                <Name>ERASE_NAMED_OBJECT</Name>
                <Frequency>1</Frequency>
                <NamedObject>
                    <object_id>{name}</object_id>
                </NamedObject>
            </elem>
        ''')
        
    def stout_smear(
        self, 
        N, 
        peram, 
        name:str='gauge', 
        orthog_dir:str = '3',
        gauge_id:str='default_gauge_field'
        ):
        print(f'''
            <elem>
                <Name>APPLY_FERM_STATE</Name>
                <Frequency>1</Frequency>
                <Param>
                    <version>1</version>
                    <FermState>
                    <Name>STOUT_FERM_STATE</Name>
                    <rho>{peram}</rho>
                    <n_smear>{N}</n_smear>
                    <orthog_dir>{orthog_dir}</orthog_dir>
                    <FermionBC>
                        <FermBC>SIMPLE_FERMBC</FermBC>
                        <boundary>1 1 1 -1</boundary>
                    </FermionBC>
                    </FermState>
                </Param>
                <NamedObject>
                    <gauge_id>{gauge_id}</gauge_id>
                    <output_id>{name}</output_id>
                </NamedObject>
            </elem>
            ''')
        self.gauge = name
        
        
    def hpy_smear(
        self,
        N, 
        alpha1, alpha2, alpha3, 
        name:str='gauge', 
        gauge_id:str='default_gauge_field'
        ):
        print(f'''
            <elem>
                <Name>LINK_SMEAR</Name>
                <Frequency>1</Frequency>
                <Param>
                    <version>5</version>
                    <LinkSmearingType>HYP_SMEAR</LinkSmearingType>
                    <alpha1>{alpha1}</alpha1>
                    <alpha2>{alpha2}</alpha2>
                    <alpha3>{alpha3}</alpha3>
                    <num_smear>{N}</num_smear>
                    <no_smear_dir>-1</no_smear_dir>  
                    <BlkMax>100</BlkMax>
                    <BlkAccu>1.0e-5</BlkAccu>
                </Param>
                <NamedObject>
                    <gauge_id>{gauge_id}</gauge_id>
                    <linksmear_id>{name}</linksmear_id>
                </NamedObject>
            </elem>      
            ''')
        self.gauge = name
        
        
    def Coulomb_gauge_fix(
        self, 
        name:str='coulomb_fix_gauge', 
        gauge_id:str='default_gauge_field'
        ):
        
        print(f'''
            <elem>
                <!-- Coulomb gauge fix -->
                <Name>COULOMB_GAUGEFIX</Name>
                <Frequency>1</Frequency>
                <Param>
                    <version>1</version>
                    <GFAccu>1.0e-7</GFAccu>
                    <GFMax>10000</GFMax>
                    <OrDo>false</OrDo>
                    <OrPara>1.0</OrPara>
                    <j_decay>3</j_decay>
                </Param>
                <NamedObject>
                    <gauge_id>{gauge_id}</gauge_id>
                    <gfix_id>{name}</gfix_id>
                    <gauge_rot_id>gauge_rot</gauge_rot_id>
                </NamedObject>
            </elem>
            ''')
        self.gauge = name
        
        
    def point_source(
        self, 
        t_srce:list=[0,0,0,0], 
        name:str='source',
        gauge:str='default_gauge_field',
        ):
        t1,t2,t3,t4 = t_srce
        print(f'''
            <elem>
                <Name>MAKE_SOURCE</Name>
                <Frequency>1</Frequency>
                <Param>
                    <version>6</version>
                    <Source>
                    <version>1</version>
                    <SourceType>POINT_SOURCE</SourceType>
                    <j_decay>3</j_decay>
                    <t_srce>{t1} {t2} {t3} {t4}</t_srce>
                    <Displacement>
                        <version>1</version>
                        <DisplacementType>NONE</DisplacementType>
                    </Displacement>
                    </Source>
                </Param>
                <NamedObject>
                    <gauge_id>{gauge}</gauge_id>
                    <source_id>{name}</source_id>
                </NamedObject>
            </elem>
            ''')
    def wall_source(
        self, 
        name:str='source',
        gauge:str='default_gauge_field',
        ):
        print(f'''
            <elem>
                <Name>MAKE_SOURCE</Name>
                <Frequency>1</Frequency>
                <Param>
                    <version>6</version>
                    <Source>
                    <version>1</version>
                    <SourceType>WALL_SOURCE</SourceType>
                    <j_decay>3</j_decay>
                    <t_source>0</t_source>
                    </Source>
                </Param>
                <NamedObject>
                    <gauge_id>{gauge}</gauge_id>
                    <source_id>{name}</source_id>
                </NamedObject>
            </elem>
              ''')
    def mom_source(
        self, 
        grid:list,
        t_srce:list=[0,0,0,0], 
        ini_mom:list=[0,0,0,0], 
        smear_mom:list=[0,0,0,0], 
        name:str='source',
        gauge:str='default_gauge_field',
        ):
        t1,t2,t3,t4 = t_srce
        mom1,mom2,mom3,mom4 = ini_mom
        smom1,smom2,smom3,smom4 = smear_mom
        gx,gy,gz,gt = grid
        print(f'''
            <elem>
                <Name>MAKE_SOURCE</Name>
                <Frequency>1</Frequency>
                <Param>
                <version>6</version>
                <Source>
                    <version>1</version>
                    <SourceType>MOM_GRID_SOURCE</SourceType>
                    <j_decay>3</j_decay>
                    <t_srce>{t1} {t2} {t3} {t4}</t_srce>
                    <grid>{gx} {gy} {gz} {gt}</grid>
                    <ini_mom>{mom1} {mom2} {mom3} {mom4}</ini_mom>
                    <SmearingParam>
                    <wvf_kind>MOM_GAUSSIAN</wvf_kind>
                    <wvf_param>1</wvf_param>
                    <wvfIntPar>1</wvfIntPar>
                    <mom>{smom1} {smom2} {smom3} {smom4}</mom>    
                    <no_smear_dir>3</no_smear_dir>
                    <qudaSmearingP>false</qudaSmearingP>
                    </SmearingParam>
                </Source>
                </Param>
                <NamedObject>
                <gauge_id>{gauge}</gauge_id>
                <source_id>{name}</source_id>
                </NamedObject>
            </elem>
              ''')
    def propagator(
        self, 
        quark_mass:str,
        clovcoeff:str,
        name:str='prop', 
        source:str='source', 
        invert:str='clover_invert',
        blocking1:list[int]=[2,2,2,2],
        blocking2:list[int]=[1,1,1,1],
        gauge:str='default_gauge_field',
        ):
        self.clovcoeff = clovcoeff
        self.quark_mass = quark_mass
            
        def clover_invert(self):
            b11,b12,b13,b14 = blocking1
            b21,b22,b23,b24 = blocking2
            print(f'''
                <InvertParam>
                    <invType>QUDA_MULTIGRID_CLOVER_INVERTER</invType>
                    <MULTIGRIDParams>
                    <RelaxationOmegaMG>1.0</RelaxationOmegaMG>
                    <RelaxationOmegaOuter>1.0</RelaxationOmegaOuter>
                    <CheckMultigridSetup>true</CheckMultigridSetup>
                    <Residual>1.0e-1</Residual>
                    <MaxIterations>12</MaxIterations>
                    <Verbosity>true</Verbosity>
                    <Precision>SINGLE</Precision>
                    <Reconstruct>RECONS_12</Reconstruct>
                    <NullVectors>24 32</NullVectors>
                    <GenerateNullspace>true</GenerateNullspace>
                    <GenerateAllLevels>true</GenerateAllLevels>
                    <CheckMultigridSetup>true</CheckMultigridSetup>
                    <CycleType>MG_RECURSIVE</CycleType>
                    <Pre-SmootherApplications>0 0</Pre-SmootherApplications>
                    <Post-SmootherApplications>8 8</Post-SmootherApplications>
                    <SchwarzType>ADDITIVE_SCHWARZ</SchwarzType>
                    <Blocking>
                        <elem>{b11} {b12} {b13} {b14}</elem>
                        <elem>{b21} {b22} {b23} {b24}</elem>
                    </Blocking>
                    </MULTIGRIDParams>
                    <SubspaceID>quda_mg_subspace</SubspaceID>
                    <ThresholdCount>500</ThresholdCount>
                    <MaxIter>10000</MaxIter>
                    <CloverParams>
                    <Mass>{self.quark_mass}</Mass>
                    <clovCoeff>{self.clovcoeff}</clovCoeff>
                    <AnisoParam>
                    <anisoP>false</anisoP>
                    <t_dir>3</t_dir>
                    <xi_0>1.0</xi_0>
                    <nu>1</nu>
                    </AnisoParam>
                    <FermionBC>
                        <FermBC>SIMPLE_FERMBC</FermBC>
                        <boundary>1 1 1 -1</boundary>
                    </FermionBC>
                    </CloverParams>
                    <RsdTarget>1e-8</RsdTarget>
                    <Delta>0.1</Delta>
                    <RsdToleranceFactor>1</RsdToleranceFactor>
                    <SilentFail>false</SilentFail>
                    <AntiPeriodicT>true</AntiPeriodicT>
                    <SolverType>GCR</SolverType>
                    <Verbose>false</Verbose>
                    <AsymmetricLinop>true</AsymmetricLinop>
                    <CudaReconstruct>RECONS_12</CudaReconstruct>
                    <CudaSloppyPrecision>SINGLE</CudaSloppyPrecision>
                    <CudaSloppyReconstruct>RECONS_12</CudaSloppyReconstruct>
                </InvertParam>
                ''')
        
        def cg_invert(self):
            print(f'''
                <InvertParam>
                <invType>CG_INVERTER</invType>
                <RsdCG>1.0e-12</RsdCG>
                <MaxCG>100000</MaxCG>
                </InvertParam>
                ''')    
            
        print(f'''
            <elem>
                <Name>PROPAGATOR</Name>
                <Frequency>1</Frequency>
                <Param>
                    <version>10</version>
                    <quarkSpinType>FULL</quarkSpinType>
                    <obsvP>true</obsvP>
                    <numRetries>1</numRetries>
                    <FermionAction>          
                        <FermAct>CLOVER</FermAct>
                        <Mass>{self.quark_mass}</Mass>
                        <clovCoeff>{self.clovcoeff}</clovCoeff>
                        <FermState>
                        <Name>STOUT_FERM_STATE</Name>
                        <rho>0.125</rho>
                        <n_smear>1</n_smear>
                        <orthog_dir>-1</orthog_dir>
                        <FermionBC>
                            <FermBC>SIMPLE_FERMBC</FermBC>
                            <boundary>1 1 1 -1</boundary>
                        </FermionBC>
                        </FermState>
                    </FermionAction>''')
        eval(invert)(self)
        print(f'''
                </Param>
                <NamedObject>
                    <gauge_id>{gauge}</gauge_id>
                    <source_id>{source}</source_id>
                    <prop_id>{name}</prop_id>
                </NamedObject>
            </elem>''')
        
        
        
    def point_sink_smear(
        self, 
        name:str='smeared_prop',  
        type:str='POINT_SINK',
        prop:str='prop',
        gauge:str='default_gauge_field',
        ):
        print(f'''
            <elem>
                <Name>SINK_SMEAR</Name>
                <Frequency>1</Frequency>
                <Param>
                    <version>5</version>
                    <Sink>
                    <version>2</version>
                    <SinkType>{type}</SinkType>
                    <j_decay>3</j_decay>
                    <Displacement>
                        <version>1</version>
                        <DisplacementType>NONE</DisplacementType>
                    </Displacement>
                    </Sink>
                </Param>
                <NamedObject>
                    <gauge_id>{gauge}</gauge_id>
                    <prop_id>{prop}</prop_id>
                    <smeared_prop_id>{name}</smeared_prop_id>
                </NamedObject>
            </elem>
                ''')
    def shell_sink_smear(
        self,
        name:str='smeared_prop',  
        prop:str='prop',
        gauge:str='default_gauge_field',
        ):
        
            
        print(f'''
            <elem>
                <Name>SINK_SMEAR</Name>
                <Frequency>1</Frequency>
                <Param>
                    <version>5</version>
                    <Sink>
                    <version>1</version>
                    <SinkType>SHELL_SINK</SinkType>
                    <j_decay>3</j_decay>
                    <SmearingParam>
                        <wvf_kind>MOM_GAUSSIAN</wvf_kind>
                        <wvf_param>0</wvf_param>
                        <wvfIntPar>0</wvfIntPar>
                        <no_smear_dir>3</no_smear_dir>
                        <mom>0 0 0 0</mom>
                    </SmearingParam>
                    <Displacement>
                        <version>1</version>
                        <DisplacementType>NONE</DisplacementType>
                    </Displacement>
                    </Sink>
                </Param>
                <NamedObject>
                    <gauge_id>{gauge}</gauge_id>
                    <prop_id>{prop}</prop_id>
                    <smeared_prop_id>{name}</smeared_prop_id>
                </NamedObject>
            </elem>
              ''')
        
    def HADRON_SPECTRUM_v2(
        self, 
        hadron_list, 
        mom_list:str=505050,  
        smeared_prop:str='smeared_prop',
        gauge:str='default_gauge_field',
        ):
            
        mom = int(mom_list)
        self.Px = 50 - mom%100
        self.Py = 50 - (mom//100)%100
        self.Pz = 50 - (mom//10000)%100
        print(f'''
            <elem>
                <annotation>
                    Compute the hadron spectrum.
                    This version is a clone of the one below, however the xml output is written within the same chroma output file.
                </annotation>
                <Name>HADRON_SPECTRUM_v2</Name>
                <Frequency>1</Frequency>
                <Param>
                    <version>1</version>
                    <hadron_list>{hadron_list}</hadron_list>
                    <mom_list>{mom_list}</mom_list>
                    <prj_type>0</prj_type>
                    <cfg_serial>{self.conf_id}</cfg_serial>
                    <avg_equiv_mom>false</avg_equiv_mom>
                    <time_rev>false</time_rev>
                    <translate>false</translate>
                </Param>
                <NamedObject>
                    <gauge_id>{gauge}</gauge_id>    
                    <sink_pairs>
                        <elem>
                            <first_id>{smeared_prop}</first_id>
                            <second_id>{smeared_prop}</second_id>
                        </elem>
                    </sink_pairs>
                </NamedObject>
                <output>{self.out_path}{self.hadron}_2pt_Px{self.Px}Py{self.Py}Pz{self.Pz}_ENV-1_conf{self.conf_id}_tsep-1_mass{self.quark_mass}.iog</output>
            </elem>
                ''')
    def seqsource_fast(
        self, 
        multi_tSinks, 
        SeqSourceType, 
        sink_mom:list[int]=[0,0,0], 
        sink_type:str='POINT_SINK',
        smeared_prop:str='smeared_prop',
        name:str='seq_source',
        Flavor:str='U',
        gauge:str='default_gauge_field',
        ):
        
        self.multi_tSinks = multi_tSinks
        mom1,mom2,mom3=sink_mom
        
        if Flavor == 'U' and self.hadron == 'nucleon':
            N = 2
        else:
            N = 1
            
        print(f'''
            <elem>
                <Name>SEQSOURCE_FAST</Name>
                <SmearedProps>true</SmearedProps>
                <multi_tSinks> {self.multi_tSinks} </multi_tSinks>
                <Frequency>1</Frequency>
                <Param>
                    <version>2</version>
                    <SeqSource>
                    <version>1</version>
                    <SeqSourceType>{SeqSourceType}</SeqSourceType>
                    <j_decay>3</j_decay>
                    <t_sink>0</t_sink>
                    <sink_mom>{mom1} {mom2} {mom3}</sink_mom>
                    </SeqSource>
                </Param>
                <PropSink>
                    <version>5</version>
                    <Sink>
                    <version>2</version>
                    <SinkType>{sink_type}</SinkType>
                    <j_decay>3</j_decay>
                    <Displacement>  
                        <version>1</version>
                        <DisplacementType>NONE</DisplacementType>
                    </Displacement>
                    </Sink>
                </PropSink>
                <NamedObject>''')
        print(f'''\
                    <prop_ids>''')
        for i in range(N):
            print(f'''\
                        <elem>{smeared_prop}</elem>''')
                
        print(f'''\
                    </prop_ids>
                    <seqsource_id>
                        <elem>{name}</elem>
                    </seqsource_id>
                    <gauge_id>{gauge}</gauge_id>
                </NamedObject>
            </elem>
        ''')
    def building_block(
        self,
        mom2_max:str='0',
        links_max:str='0',
        links_dir:str='2',
        frwd_prop_id:str='prop',
        bkwd_prop_id:str='seq_prop',
        Flavor:str='U',
        gauge='default_gauge_field',
        use_sink_offset = 'false',
        canonical = 'false',
        time_reverse = 'false',
        translate = 'false',
        conserved = 'false',
        use_gpu = 'false',
        use_cpu = 'false',
        ):
            
        print(f'''
            <elem>
            <annotation>
                "a_0" just indicates the Gamma =1; therefore the output gamma is the corret ones int eh chroma gamma index
            </annotation>
            <Name>BUILDING_BLOCKS_IOG</Name>
            <Frequency>1</Frequency>
            <Param>
            <version>6</version>
            <cfg_serial>{self.conf_id}</cfg_serial>
            <use_sink_offset>{use_sink_offset}</use_sink_offset>
            <mom2_max>{mom2_max}</mom2_max>
            <links_max>{links_max}</links_max>
            <links_dir>{links_dir}</links_dir>
            <canonical>{canonical}</canonical>
            <time_reverse>{time_reverse}</time_reverse>
            <translate>{translate}</translate>
            <conserved>{conserved}</conserved>
            <use_gpu>{use_gpu}</use_gpu>
            <use_cpu>{use_cpu}</use_cpu>
            </Param>
            <BuildingBlocks>
            <OutFileName>{self.out_path}output/{self.hadron}_3pt_Px{self.Px}Py{self.Py}Pz{self.Pz}_ENV-1_conf{self.conf_id}_tsep{self.multi_tSinks}_mass{self.quark_mass}_linkdir{links_dir}_linkmax{links_max}.out</OutFileName>
            <GaugeId>{gauge}</GaugeId>
            <FrwdPropId>{frwd_prop_id}</FrwdPropId>
            <BkwdProps>
                <elem>
                <BkwdPropId>{bkwd_prop_id}</BkwdPropId>          
                <BkwdPropG5Format>G5_B_G5</BkwdPropG5Format>
                <GammaInsertion>0</GammaInsertion>
                <Flavor>{Flavor}</Flavor>
                <BBFileNamePattern>{self.out_path}{self.hadron}_{Flavor}_3pt_Px{self.Px}Py{self.Py}Pz{self.Pz}_ENV-1_conf{self.conf_id}_tsep{self.multi_tSinks}_mass{self.quark_mass}_linkdir{links_dir}_linkmax{links_max}_conserved_{conserved}.iog</BBFileNamePattern>
                <BBFileNamePattern_gpu>{self.out_path}{self.hadron}_{Flavor}_3pt_Px{self.Px}Py{self.Py}Pz{self.Pz}_ENV-1_conf{self.conf_id}_tsep{self.multi_tSinks}_mass{self.quark_mass}_linkdir{links_dir}_linkmax{links_max}_conserved_{conserved}.iog</BBFileNamePattern_gpu>
                </elem>
            </BkwdProps>
            </BuildingBlocks>
            <xml_file>{self.out_path}xml/{self.hadron}_3pt_Px{self.Px}Py{self.Py}Pz{self.Pz}_ENV-1_conf{self.conf_id}_tsep{self.multi_tSinks}_mass{self.quark_mass}_linkdir{links_dir}_linkmax{links_max}.xml</xml_file>
        </elem>
              ''')
    def end(self):
        print(f'''
            </InlineMeasurements>    
            
            <nrow>{self.Nx} {self.Nx} {self.Nx} {self.Nt}</nrow>
            
        </Param>
        <RNG>
            <Seed>
            <elem>11</elem>
            <elem>11</elem>
            <elem>11</elem>
            <elem>0</elem>
            </Seed>
        </RNG>
        <Cfg>
            <cfg_type>SCIDAC</cfg_type>
            <cfg_file>{self.conf_dir}{self.conf_id}.lime</cfg_file>
            <parallel_io>true</parallel_io>
        </Cfg>
    </chroma>
        '''
    )