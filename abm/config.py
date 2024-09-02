ENV_SETTING = {
    'off_peak_price_period': [22, 23, 0, 1, 2, 3, 4, 5],           # 低谷电价时段
    'mid_peak_price_period': [6, 7, 11, 12, 13, 14, 15, 16, 17, 21],  # 平峰电价时段
    'peak_price_period': [8, 9, 10, 18, 19, 20],                 # 高峰电价时段
    'off_peak_prob': 0.015,                                  # 低谷时段选择充电概率（每一步）
    'mid_peak_prob': 0.015,                                 # 平峰时段选择充电概率（每一步）
    'high_peak_prob': 0.01,                                  # 高峰时段选择充电概率（每一步）
    'charge_willings': [0.6,  # 0-1点      # 24小时的充电意愿，根据出行需求和时段调整
                        0.4,  # 1-2点
                        0.2,  # 2-3点
                        0.1,  # 3-4点
                        0.1,  # 4-5点
                        0.2,  # 5-6点
                        0.4,  # 6-7点
                        0.6,  # 7-8点
                        0.8,  # 8-9点
                        1,  # 9-10点
                        1,  # 10-11点
                        1,  # 11-12点
                        1,  # 12-13点
                        1,  # 13-14点
                        1,  # 14-15点
                        1,  # 15-16点
                        1,  # 16-17点
                        0.8,  # 17-18点
                        0.6,  # 18-19点
                        0.7,  # 19-20点
                        0.8,  # 20-21点
                        0.9,  # 21-22点
                        1,  # 22-23点
                        0.8],# 23-24点
    'preference_weights': {"price": 0.5, "distance": 0.3, "capacity": 0.2}

}

SIM_SETTING = {
    'step_length': 5,                                                   # 仿真步长
    'simulation_days': 7,                                               # 仿真天数
    'use_road_network': False,                                           # 是否使用路网
    'car_type_per_list': [0.6, 0.4],
    # 电动车电池容量分组
    'car_power_distribution_list': [[17, 33, 38, 52, 77], [37, 48, 53, 70, 95]],
    'low_battery_fat_c': 0.02,                                          #
    'low_battery_fat_loc': -857.88,
    'low_battery_fat_scale': 906.74,
    'mid_battery_fat_c': 0.16,
    'mid_battery_fat_loc': -88.91,
    'mid_battery_fat_scale': 133.14,
    'high_battery_fat_c': 0.12,
    'high_battery_fat_loc': -129.47,
    'high_battery_fat_scale': 172.34,
    'consumption_rate': 20,                                             # 电动车电量消耗速率
    'charge_speed_car': 60,                                              # 电动车充电速率
    'travel_speed': 40                                                   # 电动车行驶速率

}

CAR_SETTING = {
    'car_speed': 60,                                                     # 电动车行驶速率
    # 充电站选择方法,1:最大容量,2:最便宜
    'station_selection_method': 3,
    'ratio_of_operational_vehicles': 0,                                   # 营运车比例
}

CACHE_SETTING = {
    'cache_save_dir': './cache',                                          # 路网缓存文件保存路径
    'cache_road_path': './cache/route_dict.pkl',                         # 路网缓存地址
    'cache_road_path_runtime': './cache/route_dict_runtime.pkl',                  # 运行时路网缓存地址
    'shanghai_road': './data/shanghai_road2.graphml'                      # 上海路网结构图
}

INIT_SETTING = {
    'parallel': False,                                                  # 初始化是否并行
}

OUTPUT_SETTING = {
    'output_dir': './output'                                            # 输出文件保存路径
}
