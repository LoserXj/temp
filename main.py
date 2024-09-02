import pandas as pd
import numpy as np
import geopandas as gpd
import transbigdata as tbd
import warnings
import ast 
import numpy as np
from scipy.spatial import KDTree
import abm.config as config
import imp
import random 
from pandarallel import pandarallel
import abm.CarAgent as CarAgent
import abm.ChargingStationAgent as ChargingstationAgent
import abm.Environment as Environment 
import imp
from scipy.stats import fatiguelife
import osmnx as ox
import pickle 
import abm.config as config
import os
from pathlib import Path
imp.reload(CarAgent)
imp.reload(ChargingstationAgent)
imp.reload(Environment)
pandarallel.initialize(progress_bar=True,nb_workers=10)
imp.reload(config)
warnings.filterwarnings('ignore')

def read_traj(traj_path):
        """
        读取轨迹数据，并进行预处理。
        
        参数:
        - traj_path: 字符串，表示轨迹数据文件的路径。
        
        返回值:
        - 返回两个值，第一个是轨迹数据的起始时间，第二个是预处理后的轨迹数据的DataFrame。
        
        详细说明:
        - 使用pandas库的read_csv方法读取轨迹数据文件。
        - 删除含有NaN值的行，以确保数据的完整性。
        - 根据配置文件中是否启用并行处理，使用apply或parallel_apply方法来设置'wether_home'和'wether_work'字段的值。
        这两个字段表示轨迹是否表示回家或工作的行程。
        - 删除原始的'type'列，因为它已经被转换为'wether_home'和'wether_work'字段。
        - 将'time'列转换为datetime类型，并计算每条轨迹相对于最小时间的偏移量（以分钟为单位）。
        - 移除所有在一天的开始时刻（即每小时的0分0秒）开始的轨迹，这些轨迹可能不完整。
        - 移除所有只出现一次的用户轨迹，这些用户可能只是短暂出现在轨迹数据中。
        - 返回轨迹数据的起始时间和预处理后的轨迹数据的DataFrame。
        """
        input_data = pd.read_csv(traj_path)
        input_data = input_data.dropna()
        if config.INIT_SETTING['parallel']:
            input_data['wether_home'] = input_data['type'].parallel_apply(lambda x: 1 if x=='H_0' else 0)
            input_data['wether_work'] = input_data['type'].parallel_apply(lambda x: 1 if x=='W_0' else 0)
        else:
            input_data['wether_home'] = input_data['type'].apply(lambda x: 1 if x=='H_0' else 0)
            input_data['wether_work'] = input_data['type'].apply(lambda x: 1 if x=='W_0' else 0) 
        input_data.drop(columns=['type'],inplace=True)
        input_data['time'] = pd.to_datetime(input_data['time'])
        starttime = input_data['time'].min()
        input_data = input_data[input_data['time']>=starttime]
        input_data['timestamp'] = (input_data['time']-starttime).dt.total_seconds()/60
        input_data = input_data[~((input_data['timestamp']%(24*60)==0)&(input_data['timestamp']/(24*60)>0))]
        uid_counts = input_data['uid'].value_counts()
        traj_stay_home = uid_counts[uid_counts == 1].index.tolist()
        input_data = input_data[~input_data['uid'].isin(traj_stay_home)]
        return starttime, input_data

def read_station(home_private_pile_prob, input_data, ev_station=pd.DataFrame(), public_station = pd.DataFrame()):
    """
    读取和处理充电站数据，包括私有充电桩和公共充电站。
    
    参数:
    - home_private_pile_prob: 浮点数，表示私有充电桩拥有概率。
    - input_data: DataFrame，包含轨迹数据。
    - ev_station: DataFrame，包含电动车辆的充电站数据，默认为空DataFrame。
    - public_station: DataFrame，包含公共充电站数据，默认为空DataFrame。
    
    返回值:
    - 返回两个值，第一个是合并后的充电站DataFrame，第二个是更新后的轨迹数据DataFrame。
    
    详细说明:
    - 该方法首先根据概率home_private_pile_prob为轨迹数据中的每个家位置生成私有充电桩。
    - 然后，将私有充电桩的信息添加到轨迹数据中，并处理公共充电站和电动车辆充电站的数据。
    - 最后，将所有类型的充电站合并为一个DataFrame，并返回这个DataFrame以及更新后的轨迹数据。
    """
    ## 初始化私桩
    traj_data = input_data.copy()
    uid_df = traj_data[traj_data['wether_home']==1]['uid'].drop_duplicates()
    uid_df = pd.DataFrame(uid_df)
    uid_df.reset_index(drop=True,inplace=True)
    home_private_pile_series = [random.choices([1,0],weights=[home_private_pile_prob,1-home_private_pile_prob],k=len(uid_df))]
    home_private_pile_series = list(home_private_pile_series[0])
    uid_df['private_pile'] = home_private_pile_series
    # print(uid_df['private_pile'].value_counts()[1],(home_private_pile_series.count(1)))
    traj_data = traj_data.merge(uid_df,on=['uid'],how='left')
    traj_data['private_pile'] = traj_data['private_pile'].fillna(0)
    private_pile_station = traj_data[((traj_data['private_pile']==1)&(traj_data['wether_home']==1))]
    uid_traj_private_pile = traj_data[traj_data['private_pile']==1]['uid'].drop_duplicates()
    uid_station_private = private_pile_station['uid'].drop_duplicates()
    uid_not_in_station = set(uid_traj_private_pile) - set(uid_station_private)
    uid_list = list(uid_not_in_station)
    for u in uid_list:
        traj_data[traj_data['uid']==u]['private_pile'] = 0
    if len(private_pile_station) != 0:
        private_pile_station.drop_duplicates(subset=['uid'],inplace=True)
        private_pile_station = private_pile_station[['uid','lon','lat']]
        private_pile_station['stationId'] = private_pile_station.apply(lambda r: str(int(r['uid']))+'_private',axis=1)
        private_pile_station['capacity'] = 1
        private_pile_station['charge_speed_station'] = 7
        private_pile_station['unit_price'] = 1.77
        private_pile_station['unit_price'] = private_pile_station['unit_price'].apply(lambda x: str([x for i in range(24)]))
        private_pile_station.rename(columns={'lon':'stationLon','lat':"stationLat",'uid':'owner'}, inplace=True)
        private_pile_station['station_type'] = 'private'
    ## 读取ev_station
    if len(ev_station) > 0:
        ev_station['station_type'] = 'public'
        ev_station['owner'] = -1
        
    if len(public_station) > 0:
        public_station['station_type'] = 'public'
        public_station['owner'] = -1
        public_station['charge_speed_station'] = public_station['charge_speed_station'].apply(lambda x: x if x>0 else 7.0)
        # public_station.drop(columns=['Unnamed: 0'],inplace=True)
    chargstations = pd.concat([ev_station,public_station,private_pile_station], axis=0)
    # bounds = (120.85, 30.67, 122.24, 31.87)
    # chargstations = chargstations[(chargstations['stationLon']>bounds[0])&(chargstations['stationLon']<bounds[2])
    #                               &(chargstations['stationLat']>bounds[1])&(chargstations['stationLat']<bounds[3])]
    return chargstations, traj_data

def initialize_station(chargstations,step_length):
    """
    初始化仿真环境中的充电站代理。
    
    参数:
    - chargstations: DataFrame，包含充电站的数据信息。
    - step_length: 整数，表示仿真中每一步的时间长度（分钟）。
    
    返回值:
    - 返回一个字典，键为充电站的ID，值为充电站代理对象。
    
    详细说明:
    - 该方法遍历充电站数据框chargstations中的每一行，为每个充电站创建一个充电站代理对象。
    - 充电站代理对象的属性包括充电站的ID、经纬度、最大容量、最大充电速率、每一步的时间长度、单位价格、充电站类型和所有者。
    - 最后，将创建的充电站代理对象存储在一个字典中，字典的键为充电站的ID，便于后续的仿真操作。
    """
    # 初始化充电站
    print('初始化充电站中...')
    station_agent_dict = {}
    for i in range(len(chargstations)):
        r = chargstations.iloc[i]
        station_agent_dict[r['stationId']] = ChargingstationAgent.ChargingStationAgent(
            station_id=r['stationId'],    # 充电站id
            lon=r['stationLon'],               # 经度
            lat=r['stationLat'],                # 纬度
            max_capacity=r['capacity'],              # 最大容量
            charge_speed_station=(r['charge_speed_station']),      # 该充电站最大充电速率 (kW)
            step_length=step_length,      # 每一步的时间长度（分钟）
            unit_price=ast.literal_eval(r['unit_price']),
            station_type=r['station_type'],
            station_owner=r['owner']
        )
    print('初始化充电站完成，共计', len(station_agent_dict), '个')
    return station_agent_dict


def initialize_car(input_data,step_length,car_num = None, car_type_per_list = None, car_power_distribution_list = None):
    """
    初始化仿真环境中的车辆代理。
    
    参数:
    - input_data: DataFrame，包含车辆的初始数据。
    - step_length: 整数，表示仿真中每一步的时间长度（分钟）。
    - car_num: 整数，表示要初始化的车辆数量。如果为None，则初始化所有车辆。
    - car_type_per_list: 列表，包含每种车辆类型的占比。
    - car_power_distribution_list: 列表，包含每种电池电量分布的区间。
    
    返回值:
    - 返回一个字典，键为车辆的ID，值为车辆代理对象。
    
    详细说明:
    - 该方法首先根据提供的参数设置车辆的类型和电池电量分布。
    - 然后，为每辆车创建一个车辆代理对象，包括车辆的ID、电池电量、初始位置、行程链等属性。
    - 如果指定了car_num，将从数据中随机抽样指定数量的车辆进行初始化。
    - 最后，将创建的车辆代理对象存储在一个字典中，字典的键为车辆的ID，便于后续的仿真操作。
    """
    print('初始化车辆')
    car_agent_dict = {}
    if car_type_per_list == None:
        car_type_per_list = config.SIM_SETTING['car_type_per_list']
    if car_power_distribution_list == None:
        car_power_distribution_list = config.SIM_SETTING['car_power_distribution_list']
    ## 初始化车辆
    carids = input_data['uid'].drop_duplicates()
    car_battery = []
    for i in range(len(car_type_per_list)):
        car_type_per = car_type_per_list[i]
        car_power_distribution = car_power_distribution_list[i]
        car_battery_tmp = []
        for j in range(len(car_power_distribution)-1):
            tmp_list = np.random.randint(car_power_distribution[j],car_power_distribution[j+1],size=int((len(carids)*car_type_per/(len(car_power_distribution)-1))))
            car_battery_tmp.append(tmp_list)
        car_battery_tmp = [item for sublist in car_battery_tmp for item in sublist]
        car_battery.append(car_battery_tmp)
    
    for i in range(len(car_battery)):
        np.random.shuffle(car_battery[i])
    count = 0

    def generate_car(carid,df):
        vehicle_type = np.random.choice([i for i in range(len(car_type_per_list))],p=car_type_per_list)
        total_power = np.random.choice(car_battery[vehicle_type])
        battery_type = 0
        charge_decision_power = fatiguelife.rvs(c = config.SIM_SETTING['low_battery_fat_c'], loc = config.SIM_SETTING['low_battery_fat_loc'], 
                                                scale = config.SIM_SETTING['low_battery_fat_scale'])
        if total_power>= 39 and total_power<53:
            battery_type = 1
            charge_decision_power = fatiguelife.rvs(c = config.SIM_SETTING['mid_battery_fat_c'], loc = config.SIM_SETTING['mid_battery_fat_loc'], 
                                                scale = config.SIM_SETTING['mid_battery_fat_scale'])
        if total_power>=53:
            battery_type = 2
            charge_decision_power = fatiguelife.rvs(c = config.SIM_SETTING['high_battery_fat_c'], loc = config.SIM_SETTING['high_battery_fat_loc'], 
                                                scale = config.SIM_SETTING['high_battery_fat_scale'])
        battery = int(total_power*np.random.randint(20,100)/100)
        
        init_lon = df['lon'].iloc[0]
        init_lat = df['lat'].iloc[0]
        has_private_pile = df['private_pile'].iloc[0]
        tripchain = df[['lon', 'lat', 'timestamp','wether_home','wether_work']].rename(
            columns={'lon': 'lon', 'lat': 'lat', 'timestamp': 'time'}).to_dict(orient='records')
        car_agent = CarAgent.CarAgent(
            carid=carid,                 # 车辆id
            battery=battery,                   # 剩余电量kWh
            init_lon=init_lon,            # 初始经度
            init_lat=init_lat,             # 初始纬度
            tripchain=tripchain[1:],        # 行程链
            step_length=step_length,    # 每一步的时间长度（分钟）
            vehicle_power_type = 0,         #车辆动力类型
            vehicle_type = vehicle_type,  # 车辆类型
            total_power = total_power,   # 总共的电量
            battery_type=battery_type,    # 电池类型
            charge_decision_power=charge_decision_power,
            has_private_pile = has_private_pile,
            consumption_rate=config.SIM_SETTING['consumption_rate'],        # 耗电速率（kwh/100km）
            charge_speed_car=config.SIM_SETTING['charge_speed_car'],         # 该车支持的最大充电速率 (kW)
            travel_speed=config.SIM_SETTING['travel_speed'],            # 该车的行驶速度（km/h）

        )
        return car_agent


    #是否抽样车辆
    if car_num == None:
        car_agent_dict = input_data
    else:
        car_agent_dict = pd.merge(input_data,input_data['uid'].drop_duplicates().sample(car_num))
    if config.INIT_SETTING['parallel']:
        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=True)
        car_agent_dict = car_agent_dict.groupby(['uid']).parallel_apply(lambda df:generate_car(df['uid'].iloc[0],df)).to_dict()
    else:
        car_agent_dict = car_agent_dict.groupby(['uid']).apply(lambda df:generate_car(df['uid'].iloc[0],df)).to_dict()

    print('初始化车辆完成，共计', len(car_agent_dict), '辆')
    return car_agent_dict

def initialize_environment(car_agent_dict, station_agent_dict, step_length,  start_time, workday_prob,holiday_prob):
    """
    初始化仿真环境，包括车辆和充电站代理。
    
    参数:
    - car_agent_dict: 字典，包含车辆代理对象，键为车辆ID。
    - station_agent_dict: 字典，包含充电站代理对象，键为充电站ID。
    - step_length: 整数，表示仿真中每一步的时间长度（分钟）。
    
    返回值:
    - 返回一个EnvironmentAgent对象，代表仿真环境。
    
    详细说明:
    - 该方法首先创建一个字典agent_dict，用于存储车辆和充电站代理对象。
    - 然后将agent_dict传递给EnvironmentAgent构造函数，创建一个仿真环境对象。
    - 最后，打印环境初始化完成的消息，并返回创建的仿真环境对象。
    """
    # 初始化环境
    print('初始化环境中...')
    agent_dict = {}
    agent_dict['station'] = station_agent_dict
    agent_dict['car'] = car_agent_dict 
    environment =Environment.EnvironmentAgent(
        config.SIM_SETTING['use_road_network'], 
        agent_dict=agent_dict,
        step_length=step_length,
        start_time = start_time,
        workday_prob=workday_prob,
        holiday_prob=holiday_prob
    )
    print('初始化环境完成')
    return environment

def get_station_infos(environment, starttime):
    """
    在仿真结束后，收集并输出充电站的信息。
    
    参数:
    - environment: 仿真环境对象，包含仿真过程中的所有代理和状态信息。
    - starttime: datetime对象，表示仿真开始的时间。
    
    返回值:
    - 返回一个DataFrame，包含输出的充电站信息。
    
    详细说明:
    - 该方法首先检查配置中指定的输出目录是否存在，如果不存在则创建该目录。
    - 然后，从仿真环境中提取所有充电站的信息，并将其转换为DataFrame。
    - 接着，将时间戳转换为可读的datetime格式，并填充任何缺失的值。
    - 最后，将充电站信息保存到CSV文件中，并返回这个DataFrame。
    """
    # 仿真结束，输出结果
    output_path = config.OUTPUT_SETTING['output_dir']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 输出站点信息
    station_infos = pd.DataFrame(environment.station_info)
    station_infos['time'] = starttime+pd.to_timedelta(station_infos['timestamp'], unit='m')

    station_infos['num_current_car'] = station_infos['num_current_car'].fillna(0)
    station_infos['num_waiting_car'] = station_infos['num_waiting_car'].fillna(0)
    station_infos['num_charge_car_come'] = station_infos['num_charge_car_come'].fillna(0)
    station_infos['num_charge_car_left'] = station_infos['num_charge_car_left'].fillna(0)

    station_infos.to_csv(f'{output_path}/zcm/station_infos100000.csv', index=False)
    return station_infos

def get_car_infos(environment, starttime):
    """
    在仿真结束后，收集并输出车辆的信息。
    
    参数:
    - environment: 仿真环境对象，包含仿真过程中的所有代理和状态信息。
    - starttime: datetime对象，表示仿真开始的时间。
    
    返回值:
    - 返回一个DataFrame，包含输出的车辆信息。
    
    详细说明:
    - 该方法首先获取仿真环境中存储的车辆信息列表，并将其转换为DataFrame。
    - 然后，将时间戳转换为相对于仿真开始时间的可读datetime格式。
    - 接着，对车辆信息按照车辆ID和时间进行排序。
    - 选择需要输出的列，并保存车辆信息到CSV文件中。
    - 最后，返回包含车辆信息的DataFrame。
    """
    # 车辆信息输出
    output_path = config.OUTPUT_SETTING['output_dir']
    car_infos = pd.DataFrame(environment.car_info)
    car_infos['time'] = pd.to_timedelta(car_infos['timestamp'], unit='m')+starttime

    car_infos.sort_values(by=['carid','time'],inplace=True)

    car_infos = car_infos[['carid','lon','lat','time','soc','status','current_charge_speed']]
    car_infos.to_csv(f'{output_path}/zcm/car_infos100000.csv',index=None)
    return car_infos



def main():
    """
    主函数，运行仿真流程并计算结果的MAPE。
    
    详细说明:
    - 首先获取当前.py文件的绝对路径，并设置工作目录。
    - 读取轨迹数据，并初始化输入数据。
    - 识别电动车充电站，并进行相应的数据处理。
    - 读取私有充电桩的概率，并读取或生成充电站数据。
    - 初始化车辆和充电站代理，并创建仿真环境。
    - 运行仿真指定的步数，并收集站点和车辆信息。
    - 计算仿真结果与实际订单数据之间的MAPE，并打印精度。
    """
    # 获取当前.py文件的绝对路径
    current_file_path = Path(__file__).parent.absolute()
    # 设置工作目录
    os.chdir(current_file_path)
    starttime, input_data = read_traj('./input/synthetic_data_100000.csv') 
    # 公共桩
    public_station = pd.read_csv(r'./tmp1/abm数据/zcm_station.csv')
    workday_prob = pd.read_csv('./pred_data/100_workday.csv')
    holiday_prob = pd.read_csv('./pred_data/100_holiday.csv')
    home_private_pile_prob = 340/1310 #340万为全国私桩总量，1310万为全国新能源车总量
    chargstations, input_data = read_station(
        home_private_pile_prob=home_private_pile_prob, 
        input_data=input_data, 
        public_station=public_station
        )    
    step_length = config.SIM_SETTING['step_length']
    station_agent_dict = initialize_station(chargstations, step_length)
    car_agent_dict = initialize_car(input_data=input_data,step_length=step_length,
                                    car_num=10000
                                    )
    environment = initialize_environment(
        car_agent_dict=car_agent_dict,
        station_agent_dict=station_agent_dict,
        step_length=step_length,
        start_time = starttime,
        workday_prob = workday_prob,
        holiday_prob=holiday_prob)
        # 仿真运行
    steps = int(config.SIM_SETTING['simulation_days']*24*60/step_length)
    environment.simulate(steps)
    get_station_infos(environment=environment, starttime=starttime)
    get_car_infos(environment=environment, starttime=starttime)
    
    
    
    

if __name__ == "__main__":
    main()