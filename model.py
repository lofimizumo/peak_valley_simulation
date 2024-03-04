"""
Copyright (c) [2024] [Ye Tao@RedX]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Author: ye.tao<ye.tao@redx.com.au> 
"""
from datetime import datetime, date
import pandas as pd
import numpy as np
import streamlit as st
import glob
import cProfile
import pstats
import typing


class Battery:
    def __init__(self, max_capacity=5, **kwargs):
        self.max_capacity = max_capacity
        self.cycle_times = 30
        self.min_soc = 0.1
        self.soc = 0.1
        self.current_capacity_kw = self.max_capacity * self.soc

    def get_actual_power_delta(self, command, load_power_kw):
        cmd = command.get('command', 'Idle')
        anti_backflow = command.get('anti_backflow', True)
        power_kw = command.get('power', 0)/1000  # Convert to KW
        if cmd == 'Discharge' and anti_backflow:
            power_kw = -load_power_kw
            power_kw = max(power_kw, -2.5)
        elif cmd == 'Discharge' and not anti_backflow:
            power_kw = -power_kw
        elif cmd == 'Charge':
            power_kw = power_kw
        elif cmd == 'Idle':
            power_kw = 0
        battery_num = self.max_capacity/5  # 5KWh per battery
        power_kw = power_kw * battery_num
        delta_power_kwh = power_kw * self.cycle_times / 60
        new_capacity_kw = max(self.max_capacity*self.min_soc, min(self.current_capacity_kw +
                                                                  delta_power_kwh, self.max_capacity))
        actual_delta = new_capacity_kw - self.current_capacity_kw
        self.current_capacity_kw = new_capacity_kw
        self.soc = self.current_capacity_kw / self.max_capacity
        return {'power_delta': actual_delta, 'anti_backflow': anti_backflow}


class MockData:

    def __init__(self, file_name='friend.csv', year=2023, **kwargs):
        self.year_filename_map = {
            2022: 'qld_aemo_price_2022.csv', 2023: 'qld_price_merged.csv'}
        self.df = pd.read_csv(file_name)
        self.date_start = self.df['time'].min()
        self.date_end = self.df['time'].max()
        self.df_solar = pd.read_csv('solar_clean.csv')
        self.solar_kw = kwargs.get('solar_kw', 5)
        self._truncate_date()
        self._prepare_price(year)
        if file_name != 'friend.csv':
            self.df = self.get_simulated_amber_price(self.df)
        self._prepare_solar_data()
        self._prepare_pv_data()

    def get_simulated_amber_price(self, df):
        k = 1.15
        b = 10
        df['price'] = df['price'] / 10
        df['price'] = k * df['price'] + b
        # df['price'] = df['price'].clip(0, 300)
        return df

    @classmethod
    def read_csv_from_dir(cls, dir_name):
        file_names = glob.glob(f'{dir_name}/*.csv')
        return file_names

    def _truncate_date(self):
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df[(self.df['time'] >= self.date_start)
                          & (self.df['time'] <= self.date_end)]
        self.df = self.df.sort_values('time')
        self.df.drop('price', axis=1, inplace=True)

    def _prepare_price(self, year):
        price_filename = self.year_filename_map[year]
        price_df = pd.read_csv(price_filename)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df['date'] = price_df['date'].apply(
            lambda x: x.replace(year=2023))
        price_df.set_index('date', inplace=True)
        resampled_price = price_df['price'].resample('30T').mean()
        resampled_price = resampled_price.reset_index()
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = pd.merge(self.df, resampled_price,
                           left_on='time', right_on='date')

    def _prepare_solar_data(self):
        self.df_solar['date'] = pd.to_datetime(self.df_solar['date'])
        solar_max = self.df_solar['solar_exposure'].max()
        solar_min = self.df_solar['solar_exposure'].min()
        self.df_solar['normalized_exposure'] = (
            self.df_solar['solar_exposure'] - solar_min) / (solar_max - solar_min)
        x = np.linspace(0, 288, 289)
        mean_left = 124
        mean_right = 164
        std = 28
        gaussian_left = np.exp(-((x - mean_left) ** 2) / (std ** 2))
        gaussian_right = np.exp(-((x - mean_right) ** 2) / (std ** 2))
        self.y = 8/9*(gaussian_left + gaussian_right) * self.solar_kw * 1000
        # self.y = self.y*0.9

        np.random.seed(1234)
        # rain_distribution = np.random.uniform(0.1, 1, 289)
        # self.y = self.y * rain_distribution

    def _prepare_pv_data(self):
        # Vectorize the get_solar calculation
        self.df['date'] = self.df['time'].dt.floor(
            'D')  # Get only the date part
        self.df = self.df.merge(
            self.df_solar[['date', 'normalized_exposure']], on='date', how='left')
        self.df['minute_of_day'] = self.df['time'].dt.hour * \
            60 + self.df['time'].dt.minute
        self.df['current_pv'] = self.y[self.df['minute_of_day'].values //
                                       5] * self.df['normalized_exposure']

    def get_all_data(self):
        # Return the DataFrame instead of a generator
        return self.df[['time', 'price', 'current_pv', 'usage']].copy()

    def get_usages(self, date) -> pd.Series:
        return self.df[self.df['date'] == date][['time', 'usage']].copy()

    def get_pvs(self, date) -> pd.Series:
        return self.df[self.df['date'] == date][['time', 'current_pv']].copy()


class Simulator:

    def __init__(self, date_start='2022-01-01', date_end='2022-01-31', price_gap=10, file_name=None, is_time_mode=False,
                 time_mode_discharge_start='16:00', time_mode_discharge_end='19:00',
                 time_mode_charge_start='08:00', time_mode_charge_end='16:00',
                 is_battery_only=False,
                 solar_kw=5,
                 year=2023,
                 **kwargs):
        self.model = PeakValleyScheduler(**kwargs)
        self.battery_stats = Battery(**kwargs)
        self.mock_data = MockData(
            file_name, year=year, solar_kw=solar_kw)
        self.cost_wo_battery = []
        self.cost_w_battery = []
        self.price_gap = price_gap
        self.is_time_mode = is_time_mode
        self.is_battery_only = is_battery_only
        self.time_mode_discharge_start = datetime.strptime(
            time_mode_discharge_start, '%H:%M').time()
        self.time_mode_discharge_end = datetime.strptime(
            time_mode_discharge_end, '%H:%M').time()
        self.time_mode_charge_start = datetime.strptime(
            time_mode_charge_start, '%H:%M').time()
        self.time_mode_charge_end = datetime.strptime(
            time_mode_charge_end, '%H:%M').time()
        total_discharge_duration = (self.time_mode_discharge_end.hour - self.time_mode_discharge_start.hour) * 60 + (
            self.time_mode_discharge_end.minute - self.time_mode_discharge_start.minute)
        total_charge_duration = (self.time_mode_charge_end.hour - self.time_mode_charge_start.hour) * 60 + (
            self.time_mode_charge_end.minute - self.time_mode_charge_start.minute)
        self.discharge_power = max(
            0, min(2500*120/total_discharge_duration, 2500))
        self.charge_power = max(
            0, min(1500*120/total_charge_duration, 1500))
        self.dataset = self.mock_data.get_all_data().copy()

    def get_usages(self, date: date) -> pd.Series:
        return self.mock_data.get_usages(date)

    def get_pvs(self, date: date) -> pd.Series:
        return self.mock_data.get_pvs(date)

    def get_time_mode_command(self, current_time):
        if self.is_time_mode:
            current_time = datetime.strptime(current_time, '%H:%M').time()
            if current_time >= self.time_mode_discharge_start and current_time < self.time_mode_discharge_end:
                return {'command': 'Discharge', 'power': self.discharge_power, 'anti_backflow': False}
            if current_time >= self.time_mode_charge_start and current_time <= self.time_mode_charge_end:
                return {'command': 'Charge', 'power': self.charge_power, 'grid_charge': False}
        return {'command': 'Idle'}

    def run_simulation(self):
        # Get all mock data at once
        mock_data_df = self.dataset
        mock_data_df['price_dollar'] = mock_data_df['price'] / 100
        mock_data_df['usage_with_pv'] = mock_data_df['usage'] - \
            mock_data_df['current_pv'] / 1000
        mock_data_df['load_power_kw'] = mock_data_df['usage'] * 2

        # Initialize lists for collecting data
        battery_soc_list, battery_capacity_list, action_list, power_delta_list, max_power_feedin, high_price_list = [], [], [], [], [], []
        solar_kw_list = []
        load_list = []
        grid_with_battery_solar_list = []
        grid_without_battery_list = []
        buy_price_list = []
        sell_price_list = []
        peak_price_list = []
        anti_backflow_list = []

        # Iterating through DataFrame rows (this loop might still be necessary due to sequential battery updates)
        for index, row in mock_data_df.iterrows():
            # 1. Get current price and other inputs
            now = row['time'].strftime('%H:%M')

            # 2. Model step
            command, is_high_price, buy_price, sell_price, peak_price = self.model.step(
                row['price'], now, row['usage'],
                self.battery_stats.current_capacity_kw, row['current_pv'] / 1000, device_type="2505")
            if self.is_time_mode:
                command = self.get_time_mode_command(now)
                is_high_price = True
            if self.is_battery_only:
                is_high_price = False
                command['command'] = 'Idle'

            # 3. Update battery state
            power_delta = self.battery_stats.get_actual_power_delta(
                command, row['load_power_kw'])['power_delta']
            action = command.get('command', 'Idle')
            battery_soc_list.append(f'{self.battery_stats.soc:.2%}')
            battery_capacity_list.append(
                f'{self.battery_stats.current_capacity_kw:.2f}KWh')
            action_list.append(action)
            power_delta_list.append(power_delta)
            high_price_list.append(is_high_price)
            solar_kw_list.append(row['current_pv']/1000)
            load_list.append(row['load_power_kw'])
            grid_with_battery_solar_list.append(
                row['load_power_kw'] - row['current_pv']/1000 + 2*power_delta) # 2*power_delta because power_delta is in kwh per half hour, so we need to multiply by 2 to get the hourly power in kw
            grid_without_battery_list.append(
                row['load_power_kw'] - row['current_pv']/1000)
            buy_price_list.append(buy_price)
            sell_price_list.append(sell_price)
            peak_price_list.append(peak_price)
            anti_backflow_list.append(command.get('anti_backflow', False))

            # 4. Calculate max power feeding
            if action == 'Discharge':
                max_power_feedin.append(1 if command.get(
                    'anti_backflow', False) == False else 0)

        # Update DataFrame with results from the loop
        mock_data_df['battery_soc'] = battery_soc_list
        mock_data_df['battery_capacity_Kwh'] = battery_capacity_list
        mock_data_df['action'] = action_list
        mock_data_df['power_delta'] = power_delta_list
        mock_data_df['is_high_price'] = high_price_list
        mock_data_df['solar_kw'] = solar_kw_list
        mock_data_df['charging_discharging_power'] = [
            2*x for x in power_delta_list]
        mock_data_df['load'] = load_list
        mock_data_df['grid_with_battery_solar'] = grid_with_battery_solar_list
        mock_data_df['grid_without_battery'] = grid_without_battery_list
        mock_data_df['buy_price'] = buy_price_list
        mock_data_df['sell_price'] = sell_price_list
        mock_data_df['peak_price'] = peak_price_list
        mock_data_df['anti_backflow'] = anti_backflow_list

        # 4. Update cost
        mock_data_df['price_gap'] = mock_data_df.apply(
            lambda row: 9 if not row['is_high_price'] else max(9, self.price_gap), axis=1)
        mock_data_df['feedin_price_dollar'] = (
            mock_data_df['price'] - mock_data_df['price_gap']) / 100
        mock_data_df['feedin_price_dollar'] = mock_data_df['feedin_price_dollar'].clip(
            lower=0)
        mock_data_df['usage_with_pv_battery'] = mock_data_df['usage_with_pv'] + \
            mock_data_df['power_delta']
        mock_data_df['cost_wo_battery'] = mock_data_df['price_dollar'] * \
            mock_data_df['usage']
        mock_data_df['cost_w_battery'] = mock_data_df.apply(
            lambda row: (row['feedin_price_dollar'] if row['grid_with_battery_solar'] < 0 else row['price_dollar']) * 0.5*row['grid_with_battery_solar'], axis=1)

        # Calculate total cost and savings
        total_cost_wo_battery = mock_data_df['cost_wo_battery'].sum()
        total_cost_w_battery = mock_data_df['cost_w_battery'].sum()
        total_savings = total_cost_wo_battery - total_cost_w_battery
        total_saved_percentage = total_savings / total_cost_wo_battery * 100

        # Calculate total backflow percentage
        if len(max_power_feedin) > 0:
            total_backflow_percentage = sum(
                max_power_feedin) / len(max_power_feedin) * 100
        else:
            total_backflow_percentage = 0

        # Prepare final DataFrame
        final_df = mock_data_df[['time', 'battery_soc', 'battery_capacity_Kwh', 'price',
                                 'action', 'cost_wo_battery', 'cost_w_battery', 'power_delta', 'solar_kw', 'charging_discharging_power', 'load', 'grid_with_battery_solar', 'grid_without_battery', 'buy_price', 'sell_price', 'peak_price', 'anti_backflow']]
        final_df.rename(columns={'time': 'time', 'price': 'price',
                                 'cost_wo_battery': 'cost', 'cost_w_battery': 'cost_with_solar_bat', 'charging_discharging_power': 'battery_power (kw)',
                                 'power_delta': 'power_delta (kWh)', 'solar_kw': 'solar (kw)', 'load': 'load (kw)',
                                 'grid_with_battery_solar': 'grid_w/battery_solar (kw)', 'grid_without_battery': 'grid_w/o_battery (kw)',
                                 'buy_price': 'buy_price (c)', 'sell_price': 'sell_price (c)',
                                 'peak_price': 'peak_price (c)', 'anti_backflow': 'anti_backflow'}, inplace=True)

        return {"df": final_df, "total_saved": total_saved_percentage, "total_backflow": total_backflow_percentage, "money_made": total_savings}

    def get_power_cost(self):
        return sum(self.cost_wo_battery)

    def get_power_cost_savings(self):
        cost_original = sum(self.cost_wo_battery)
        cost_savings = sum(self.cost_w_battery) - cost_original
        return cost_savings


class PeakValleyScheduler():
    def __init__(self, buy_percentile=30, sell_percentile=65, peak_percentile=90, peak_price=200, look_back_days=2, jc_param1=30, jc_param2=50, jc_param3=30, DisChgStart2='16:05', DisChgEnd2='23:55', ChgStart1='04:00', ChgEnd1='16:00', price_gap=10, **kwargs):
        """
        Initialize the model with the given parameters.

        Parameters:
        - buy_percentile (int): The percentile value for buying decision.
        - sell_percentile (int): The percentile value for selling decision.
        - peak_percentile (int): The percentile value for peak detection.
        - peak_price (int): The price threshold for peak detection.
        - look_back_days (int): The number of days to look back for historical data.
        - jc_param1 (int): The value of fixed soft discharging (anti-backflow enabled) price threshold
        - jc_param2 (int): The value of fixed strong discharging (anti-backflow disabled) price threshold 
        """
        # Constants and Initializations
        self.BatNum = 1
        self.BatMaxCapacity = 5
        self.BatCap = self.BatNum * self.BatMaxCapacity
        self.BatChgMax = self.BatNum * 1.5
        self.BatDisMax = self.BatNum * 2.5
        self.BatSocMin = 0.1
        self.HrMin = 30 / 60
        self.SellDiscount = 0.12
        self.SpikeLevel = 300
        self.SolarCharge = 0
        self.SellBack = 0
        self.BuyPct = buy_percentile
        self.SellPct = sell_percentile
        self.PeakPct = peak_percentile
        self.PeakPrice = peak_price
        self.JCParam1 = jc_param1
        self.JCParam2 = jc_param2
        self.JCParam3 = jc_param3
        self.price_gap = price_gap
        self.LookBackBars = look_back_days * 48
        self.ChgStart1 = ChgStart1
        self.ChgEnd1 = ChgEnd1
        self.DisChgStart2 = DisChgStart2
        self.DisChgEnd2 = DisChgEnd2
        self.PeakStart = '18:00'
        self.PeakEnd = '20:00'

        self.date = None
        self.last_updated_time = None

        # Initial data containers and setup
        self.price_history = []
        self.price_history_updating = []
        self.solar = None
        self.soc = self.BatSocMin
        self.bat_cap = self.soc * self.BatCap

        # Convert start and end times to datetime.time
        self.t_chg_start1 = datetime.strptime(
            self.ChgStart1, '%H:%M').time()
        self.t_chg_end1 = datetime.strptime(
            self.ChgEnd1, '%H:%M').time()
        self.t_dis_start2 = datetime.strptime(
            self.DisChgStart2, '%H:%M').time()
        self.t_dis_end2 = datetime.strptime(
            self.DisChgEnd2, '%H:%M').time()
        self.t_peak_start = datetime.strptime(
            self.PeakStart, '%H:%M').time()
        self.t_peak_end = datetime.strptime(
            self.PeakEnd, '%H:%M').time()
        self.init_price_history()

    def init_price_history(self):
        self.price_history = [20 for i in range(self.LookBackBars)]

    def step_with_fixed_lookback_price(self, current_price, current_time, current_usage, current_soc, current_pv, device_type):

        # Update battery state
        self.bat_cap = current_soc * self.BatCap

        # Update price history every five minutes
        current_time = datetime.strptime(
            current_time, '%H:%M').time()
        if self.last_updated_time is None or current_time.minute != self.last_updated_time.minute:
            self.last_updated_time = current_time
            self.price_history_updating.append(current_price)

        if len(self.price_history_updating) > 2*self.LookBackBars:
            self.price_history_updating = self.price_history_updating[-self.LookBackBars:]
            self.price_history = self.price_history_updating.copy()

        buy_price, sell_price = np.percentile(
            self.price_history, [self.BuyPct, self.SellPct])
        peak_price = np.percentile(
            self.price_history, self.PeakPct)
        fixed_peak_price = self.PeakPrice
        is_high_price = False
        current_feedin_price = current_price - self.price_gap
        if current_price > np.percentile(self.price_history, 90):
            is_high_price = True
        else:
            current_feedin_price = current_price - 10

        command = {"command": "Idle"}

        if self._is_charging_period(current_time) and ((current_price <= buy_price) or (current_pv > current_usage)):
            power = 2500 if device_type == "5000" else 1500
            command = {'command': 'Charge', 'power': power,
                       'grid_charge': True if current_pv <= current_usage else False}

        power = 5000 if device_type == "5000" else 2500
        if self._is_discharging_period(current_time) and (current_feedin_price >= sell_price) and (current_feedin_price >= self.JCParam1):
            anti_backflow = False if current_feedin_price > np.percentile(
                self.price_history, self.PeakPct) else True
            anti_backflow = True if current_feedin_price <= self.JCParam2 else anti_backflow

            command = {'command': 'Discharge', 'power': power,
                       'anti_backflow': anti_backflow}

        if current_feedin_price > fixed_peak_price and current_pv < current_usage:
            command = {'command': 'Discharge',
                       'power': power, 'anti_backflow': False}
        return command, is_high_price, buy_price, sell_price, peak_price

    def step(self, current_price, current_time, current_usage, current_soc, current_pv, device_type):

        # Update battery state
        self.bat_cap = current_soc * self.BatCap

        # Update price history every five minutes
        current_time = datetime.strptime(
            current_time, '%H:%M').time()

        if self.last_updated_time is None or current_time.minute != self.last_updated_time.minute:
            self.last_updated_time = current_time
            self.price_history.append(current_price)

        if len(self.price_history) > self.LookBackBars:
            self.price_history.pop(0)

        buy_price, sell_price = np.percentile(
            self.price_history, [self.BuyPct, self.SellPct])
        peak_price = np.percentile(
            self.price_history, self.PeakPct)
        fixed_peak_price = self.PeakPrice
        is_high_price = False
        current_feedin_price = current_price - self.price_gap
        if current_price > np.percentile(self.price_history, 90):
            is_high_price = True
        else:
            current_feedin_price = current_price - 10

        command = {"command": "Idle"}

        if self._is_charging_period(current_time) and ((current_price <= buy_price) or (current_pv > current_usage)):
            maxpower = 2500 if device_type == "5000" else 1500
            excess_solar = 1000*(current_pv - current_usage)
            if excess_solar > 0:
                power = min(maxpower, excess_solar)
                # logging.info(
                #     f"Increase charging power due to excess solar: {excess_solar}, adjusted power: {power}")
            else:
                power = min(max((5000 - 1000*current_usage), 0), maxpower)
            command = {'command': 'Charge', 'power': power,
                       'grid_charge': True if current_pv <= current_usage else False}

        power = 5000 if device_type == "5000" else 2500
        if self._is_discharging_period(current_time) and (current_price >= sell_price) and (current_price >= self.JCParam1):
            anti_backflow = False if current_price > np.percentile(
                self.price_history, self.PeakPct) else True
            anti_backflow = True if current_price <= self.JCParam2 else anti_backflow

            command = {'command': 'Discharge', 'power': power,
                       'anti_backflow': anti_backflow}

        if current_price > fixed_peak_price and current_pv < current_usage:
            command = {'command': 'Discharge',
                       'power': power, 'anti_backflow': False}
        return command, is_high_price, buy_price, sell_price, peak_price

    def _is_charging_period(self, t):
        return t >= self.t_chg_start1 and t <= self.t_chg_end1

    def _is_discharging_period(self, t):
        # return True
        return (t >= self.t_dis_start2 and t <= self.t_dis_end2) 

    def _is_peak_period(self, t):
        return t >= self.t_peak_start and t <= self.t_peak_end

    def required_data(self):
        return ['current_price', 'current_time', 'current_usage', 'current_soc', 'current_pv']


# def find_optimal_parameters():

#     def objective(trial):
#         # Define the search space for the parameters
#         buy_percentile = trial.suggest_int('buy_percentile', 0, 30)
#         sell_percentile = trial.suggest_int('sell_percentile', 30, 100)
#         peak_percentile = trial.suggest_int('peak_percentile', 30, 100)
#         look_back_days = trial.suggest_int('look_back_days', 1, 20)

#         # Run the simulation and get the power cost savings
#         simulator = Simulator(date_start='2021-07-01', date_end='2022-03-03', buy_percentile=buy_percentile,
#                               sell_percentile=sell_percentile, peak_percentile=peak_percentile, look_back_days=look_back_days)
#         simulator.run_simulation()
#         power_cost_savings = simulator.get_power_cost_savings()

#         return power_cost_savings

#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=100)

#     best_params = study.best_params
#     best_value = study.best_value

#     print("Best Parameters:", best_params)
#     print("Best Value:", best_value)

class SimulationVisualizer:

    def __init__(self, df):
        self.df = df

    def get_resampled_data(self, start_date: date):
        start_date = pd.to_datetime(start_date)
        end_date = start_date + pd.Timedelta(days=1)
        df_day = self.df[(self.df['time'] >= start_date)
                         & (self.df['time'] < end_date)]
        if df_day.empty:
            return pd.DataFrame()
        df_30min = df_day[['time', 'price', 'battery_power (kw)']
                          ].resample('30min', on='time').median()
        return df_30min.reset_index()

    def get_discharge_percentage(self):
        discharge_days = self.df[self.df['action']
                                 == 'Discharge']['time'].dt.date.nunique()
        total_days = self.df['time'].dt.date.nunique()
        discharge_percentage = (discharge_days / total_days) * 100
        return discharge_percentage


if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    file_names = MockData.read_csv_from_dir('redx_data')
    st.set_page_config(page_title=None, page_icon=None, layout="wide",
                       initial_sidebar_state="auto", menu_items=None)
    st.title("Peak Valley Simulation Analysis")

    # Run the simulation
    with st.sidebar:
        selected_filename = st.selectbox("Device", file_names)
        st.markdown("---")
        st.write("Time Mode Settings")
        on = st.toggle('Toggle Time Mode', value=False, help="Time Mode is a mode that uses a fixed time window for charging and discharging. When the time is within the window, the battery will charge or discharge based on the price. When the time is outside the window, the battery will be idle.")
        time_mode_start = st.time_input(
            "Time Mode Discharging Start", datetime.strptime('17:00', '%H:%M').time())
        time_mode_end = st.time_input(
            "Time Mode Discharging End", datetime.strptime('19:00', '%H:%M').time())
        time_mode_charge_start = st.time_input(
            "Time Mode Charging Start", datetime.strptime('07:00', '%H:%M').time())
        time_mode_charge_end = st.time_input(
            "Time Mode Charging End", datetime.strptime('13:00', '%H:%M').time())
        st.markdown("---")
        st.write("Simulation Parameters")
        solar_on = st.toggle('Toggle Solar Only', help="Solar Only Mode")
        year = st.radio(
            "Price Year",
            [2022, 2023],
            index=1)
        solar_kw = st.slider("Solar (kW)", 0, 15, 5, help="Solar kW")
        battery_capacity = st.slider(
            "Battery Capacity (kWh)", 1, 30, 5, help="Battery Capacity in kWh")
        buy_percentile = st.slider(
            "Buy percentile (%)", 1, 100, 20, help="Start to charge battery when price is below this value")
        sell_percentile = st.slider(
            "Sell percentile (%)", 1, 100, 30, help="Start to discharge battery when price is above this value")
        peak_percentile = st.slider(
            "Peak percentile (%)", 1, 100, 70, help="Discharge w/o anti-backflow when price is above this value")
        peak_price = st.slider("Peak price (c)",  1, 1000,
                               1000, help="Daytime peak price threshold")
        high_price_gap = st.slider("High Price gap (c)",  1, 50,
                                   11, help="The gap between the feedin price and the buy/sell price when the price is high (>sell price)")
        look_back_days = st.slider(
            "Look Back Days", 1, 20, 1, help="The buy/sell price is calculated based on the historical data in the past X days")
        st.markdown("---")
        st.write("AI Mode Settings")
        discharge_window_start = st.time_input(
            "Discharging Window Start", datetime.strptime('17:00', '%H:%M').time())
        discharge_window_end = st.time_input(
            "Discharging Window End", datetime.strptime('23:55', '%H:%M').time())
        charge_window_start = st.time_input(
            "Charging Window Start", datetime.strptime('08:00', '%H:%M').time())
        charge_window_end = st.time_input(
            "Charging Window End", datetime.strptime('14:00', '%H:%M').time())
        st.markdown("---")
        st.write("Return of Investment Estimation")
        solar_cost_perKw = st.number_input(
            "Solar Cost ($/kW)", value=None, placeholder="Type in (c)")
        battery_cost_perKw = st.number_input(
            "Battery Cost ($/kWh)", value=None, placeholder="Type in (c)")
        st.markdown("---")
        st.write("Fixed Price Thresholds")
        jc_param1 = st.slider(
            "Weak Discharging Threshold", 0, 1000, 0, help="The value of fixed weak discharging (anti-backflow enabled) price threshold")
        jc_param2 = st.slider(
            "Max Discharging Threshold", 0, 1000, 0, help="The value of fixed max discharging (anti-backflow disabled) price threshold")
        jc_param3 = st.slider(
            "Charging Price Threshold", 0, 1000, 0, help="The value of fixed charging price threshold")

    simulator = Simulator(date_start='2023-01-01', date_end='2023-12-31', price_gap=high_price_gap, file_name=selected_filename,
                          buy_percentile=buy_percentile,
                          sell_percentile=sell_percentile, peak_percentile=peak_percentile, peak_price=peak_price, look_back_days=look_back_days, jc_param1=jc_param1, jc_param2=jc_param2, jc_param3=jc_param3,
                          DisChgStart2=discharge_window_start.strftime(
                              '%H:%M'),
                          DisChgEnd2=discharge_window_end.strftime('%H:%M'),
                          ChgStart1=charge_window_start.strftime('%H:%M'),
                          ChgEnd1=charge_window_end.strftime('%H:%M'),
                          is_time_mode=on,
                          is_battery_only=solar_on,
                          solar_kw=solar_kw,
                          max_capacity=battery_capacity,
                          year=year,
                          time_mode_discharge_start=time_mode_start.strftime(
                              '%H:%M'),
                          time_mode_discharge_end=time_mode_end.strftime(
                              '%H:%M'),
                          time_mode_charge_start=time_mode_charge_start.strftime(
                              '%H:%M'),
                          time_mode_charge_end=time_mode_charge_end.strftime(
                              '%H:%M')
                          )
    ret = simulator.run_simulation()
    df = ret["df"]

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.sort_stats('time').print_stats(10)
    # stats.sort_stats('cumulative').print_stats(10)

    col1, col2 = st.columns(2)
    # Display command line output using tabulate
    with col1:
        st.subheader("Cost Savings Summary")

        st.markdown(
            """
            <style>
            .big-font {
                font-size:30px !important;
                font-weight: bold;
            }
            .usage-box {
                background-color: #333333;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            .data-point {
                margin: 10px;
            }
            </style>
            """, unsafe_allow_html=True)

        # Data points
        percentage = SimulationVisualizer(df).get_discharge_percentage()
        money_made = f"{int(ret['money_made'])}"
        usage_cost = f"{ret['total_saved']:.2f}%"
        total_usage = f"{ret['total_backflow']:.2f}%"
        percent_renewables = f"{percentage:.2f}%"
        roi = f"{(solar_cost_perKw * solar_kw + battery_cost_perKw * battery_capacity)/int(ret['money_made']):.1f}" if solar_cost_perKw and battery_cost_perKw else "N/A"

        # UI layout
        col_1,  col_3, col_4 = st.columns(3)

        with col_1:
            st.markdown(
                f'<div class="usage-box data-point big-font">{usage_cost}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="usage-box data-point">TOTAL SAVINGS</div>', unsafe_allow_html=True)

        # with col_2:
        #     st.markdown(
        #         f'<div class="usage-box data-point big-font">{total_usage}</div>', unsafe_allow_html=True)
        #     st.markdown(
        #         '<div class="usage-box data-point">FEEDIN</div>', unsafe_allow_html=True)

        with col_3:
            st.markdown(
                f'<div class="usage-box data-point big-font">{percent_renewables}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="usage-box data-point">BATTERY ACTIVITY</div>', unsafe_allow_html=True)

        with col_4:
            st.markdown(
                f'<div class="usage-box data-point big-font">{money_made}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="usage-box data-point">MONEY SAVED ($)</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="usage-box data-point">Return of Investment: {roi} year</div>', unsafe_allow_html=True)

        df_discharge = df[df['action'] == 'Discharge']
        df_discharge = df_discharge[df_discharge['power_delta (kWh)'] < 0]
        hist_discharge = np.histogram(df_discharge['price'], bins=[
            0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 1000], weights=0.1*df_discharge['power_delta (kWh)'])
        df_hist = pd.DataFrame(hist_discharge)
        df_hist_inverted = df_hist.T
        df_hist_inverted.columns = ['y', 'x']
        df_hist_inverted['y'] = -df_hist_inverted['y']
        st.markdown("---")
        st.subheader("Discharge price distribution")
        st.write(
            "The following chart shows the distribution of the feedin price when the battery is discharging.")
        st.bar_chart(df_hist_inverted, x='x', y='y')
        st.markdown("---")
    with col2:
        st.subheader("Battery Discharging Distribution by Time")
        sim_visualizer = SimulationVisualizer(df)
        day = st.date_input("Please choose date", date(2023, 1, 5))
        usages = simulator.get_usages(np.datetime64(day))
        usages['usage'] = usages['usage'] *2*1000
        pvs = simulator.get_pvs(np.datetime64(day))
        pvs['current_pv'] = pvs['current_pv']

        df_30min = sim_visualizer.get_resampled_data(day)
        if not df_30min.empty:
            df_30min['time'] = pd.to_datetime(df_30min['time'])
            usages['time'] = pd.to_datetime(usages['time'])
            pvs['time'] = pd.to_datetime(pvs['time'])
            combined_data = pd.merge(
                usages, df_30min[['time', 'battery_power (kw)']], on='time', how='left')
            combined_data = pd.merge(
                combined_data, pvs[['time', 'current_pv']], on='time', how='left')
            combined_data.set_index('time', inplace=True)
            # show in watts
            combined_data['battery_power (kw)'] = combined_data['battery_power (kw)']*1000
            combined_data.rename(
                columns={'battery_power (kw)': ' power (W)', 'usage': 'load (W)', 'current_pv': 'solar power (W)'}, inplace=True)

            st.line_chart(combined_data, color=[
                          '#338F37', '#A1A12B', '#23BFA7'])
            # st.bar_chart(df_30min, x='time', y='power_delta')
            st.bar_chart(df_30min, x='time', y='price')

    st.subheader("Detailed Simulation Results")
    st.write(
        "The following table shows the simulation results on a 30-min freq from the start date to the end date.")
    multi_md_text = '''`battery` soc: the state of charge of the battery  
                        `price`: the price of the electricity  
                        `action`: the action taken by the battery  
                        `cost`: the cost of electricity without battery  
                        `cost_savings`: the cost of electricity with battery'''
    st.markdown(multi_md_text)
    st.write(df)
    # st.write(simulator.discharge_power)
    date = datetime.strptime('2023-09-17', '%Y-%m-%d').date()
