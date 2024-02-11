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


class Battery:
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.state_of_charge = 0
        self.cycle_times = 30

    def get_actual_power_delta(self, command, usage):
        cmd = command.get('command', 'Idle')
        anti_backflow = command.get('anti_backflow', True)
        power = command.get('power', 0)
        if cmd == 'Discharge' and anti_backflow:
            power = -usage*1000
        elif cmd == 'Discharge' and not anti_backflow:
            power = -power
        elif cmd == 'Charge':
            power = power
        delta_power = power * self.cycle_times / 60
        new_soc = max(0, min(self.state_of_charge +
                      delta_power, self.max_capacity))
        actual_delta = new_soc - self.state_of_charge
        self.state_of_charge = new_soc
        return {'power_delta': actual_delta, 'anti_backflow': anti_backflow}


class MockData:

    def __init__(self, file_name='friend.csv'):
        self.df = pd.read_csv(file_name)
        self.date_start = self.df['time'].min()
        self.date_end = self.df['time'].max()
        self.df_solar = pd.read_csv('solar_clean.csv')
        self._prepare_data()
        if file_name != 'friend.csv':
            self.df = self.get_simulated_amber_price(self.df)
        self._prepare_solar_data()
        self._prepare_pv_data()

    def get_simulated_amber_price(self, df):
        k = 1.15
        b = 10
        df['price'] = df['price'] / 10
        df['price'] = k * df['price'] + b
        return df

    def read_csv_from_dir(self, dir_name):
        file_names = glob.glob(f'{dir_name}/*.csv')
        return file_names

    def _prepare_data(self):
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df[(self.df['time'] >= self.date_start)
                          & (self.df['time'] <= self.date_end)]
        self.df = self.df.sort_values('time')

    def _prepare_solar_data(self):
        self.df_solar['date'] = pd.to_datetime(self.df_solar['date'])
        solar_max = self.df_solar['solar_exposure'].max()
        solar_min = self.df_solar['solar_exposure'].min()
        self.df_solar['normalized_exposure'] = (
            self.df_solar['solar_exposure'] - solar_min) / (solar_max - solar_min)
        x = np.linspace(0, 288, 289)
        mean = 144
        std = 50
        self.y = np.exp(-((x - mean) ** 2) / (std ** 2)) * 4000

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
        return self.df[['time', 'price', 'current_pv', 'usage']]


class Simulator:

    def __init__(self, date_start='2022-01-01', date_end='2022-01-31', price_gap=10, file_name=None, is_time_mode=False,
                 time_mode_discharge_start='16:00', time_mode_discharge_end='19:00',
                 time_mode_charge_start = '04:00', time_mode_charge_end = '16:00',
                 **kwargs):
        self.model = PeakValleyScheduler(**kwargs)
        self.battery_stats = Battery(max_capacity=5000)
        self.mock_data = MockData(file_name)
        self.cost_wo_battery = []
        self.cost_w_battery = []
        self.price_gap = price_gap
        self.is_time_mode = is_time_mode
        self.time_mode_discharge_start = datetime.strptime(time_mode_discharge_start, '%H:%M').time()
        self.time_mode_discharge_end = datetime.strptime(time_mode_discharge_end, '%H:%M').time()
        self.time_mode_charge_start = datetime.strptime(time_mode_charge_start, '%H:%M').time()
        self.time_mode_charge_end = datetime.strptime(time_mode_charge_end, '%H:%M').time()

    def get_time_mode_command(self, current_time):
        if self.is_time_mode:
            current_time = datetime.strptime(current_time, '%H:%M').time()
            if current_time >= self.time_mode_discharge_start and current_time <= self.time_mode_discharge_end:
                return {'command': 'Discharge', 'power': 1800, 'anti_backflow': False}
            if current_time >= self.time_mode_charge_start and current_time <= self.time_mode_charge_end:
                return {'command': 'Charge', 'power': 800, 'grid_charge': False}
        return {'command': 'Idle'}

    def run_simulation(self):
        # Get all mock data at once
        mock_data_df = self.mock_data.get_all_data().copy()
        mock_data_df['price_dollar'] = mock_data_df['price'] / 100
        mock_data_df['feedin_price_dollar'] = (
            mock_data_df['price'] - self.price_gap) / 100
        mock_data_df['usage_with_pv'] = mock_data_df['usage'] - \
            mock_data_df['current_pv'] / 1000

        # Initialize lists for collecting data
        battery_soc_list, action_list, power_delta_list, max_power_feedin, high_price_list = [], [], [], [], []

        # Iterating through DataFrame rows (this loop might still be necessary due to sequential battery updates)
        for index, row in mock_data_df.iterrows():
            # 1. Get current price and other inputs
            now = row['time'].strftime('%H:%M')

            # 2. Model step
            command, is_high_price = self.model.step(
                row['price'], now, row['usage'],
                self.battery_stats.state_of_charge, row['current_pv'] / 1000, device_type="2505")
            if self.is_time_mode: 
                command = self.get_time_mode_command(now)
                is_high_price = True

            # 3. Update battery state
            power_delta = self.battery_stats.get_actual_power_delta(
                command, row['usage'])['power_delta']
            action = command.get('command', 'Idle')
            battery_soc_list.append(self.battery_stats.state_of_charge)
            action_list.append(action)
            power_delta_list.append(power_delta)
            high_price_list.append(is_high_price)

            # 4. Calculate max power feeding
            if action == 'Discharge':
                max_power_feedin.append(1 if command.get(
                    'anti_backflow', False) == False else 0)

        # Update DataFrame with results from the loop
        mock_data_df['battery_soc'] = battery_soc_list
        mock_data_df['action'] = action_list
        mock_data_df['power_delta'] = power_delta_list
        mock_data_df['is_high_price'] = high_price_list

        # 4. Update cost
        mock_data_df['price_gap'] = mock_data_df.apply(
            lambda row: 10 if not row['is_high_price'] else self.price_gap, axis=1)
        mock_data_df['usage_with_pv_battery'] = mock_data_df['usage_with_pv'] + \
            mock_data_df['power_delta'] / 1000
        mock_data_df['cost_wo_battery'] = mock_data_df['price_dollar'] * \
            mock_data_df['usage']
        mock_data_df['cost_w_battery'] = mock_data_df.apply(
            lambda row: (row['feedin_price_dollar'] if row['usage_with_pv_battery'] < 0 else row['price_dollar']) * row['usage_with_pv_battery'], axis=1)

        # Calculate total cost and savings
        total_cost_wo_battery = mock_data_df['cost_wo_battery'].sum()
        total_cost_w_battery = mock_data_df['cost_w_battery'].sum()
        total_savings = total_cost_wo_battery - total_cost_w_battery
        total_saved_percentage = total_savings / total_cost_wo_battery * 100
        total_backflow_percentage = sum(
            max_power_feedin) / len(max_power_feedin) * 100

        # Prepare final DataFrame
        final_df = mock_data_df[['time', 'battery_soc', 'price',
                                 'action', 'cost_wo_battery', 'cost_w_battery', 'power_delta']]
        final_df.rename(columns={'time': 'time', 'price': 'price',
                        'cost_wo_battery': 'cost', 'cost_w_battery': 'cost_savings'}, inplace=True)

        return {"df": final_df, "total_saved": total_saved_percentage, "total_backflow": total_backflow_percentage, "money_made": total_savings}

    def get_power_cost(self):
        return sum(self.cost_wo_battery)

    def get_power_cost_savings(self):
        cost_original = sum(self.cost_wo_battery)
        cost_savings = sum(self.cost_w_battery) - cost_original
        return cost_savings


class PeakValleyScheduler():
    def __init__(self, buy_percentile=30, sell_percentile=65, peak_percentile=90, peak_price=200, look_back_days=2, jc_param1=30, jc_param2=50, jc_param3=30, DisChgStart2='16:05', DisChgEnd2='23:55', ChgStart1='04:00', ChgEnd1='16:00', price_gap=10):
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
        self.DisChgStart1 = '0:00'
        self.DisChgEnd1 = '04:00'
        self.PeakStart = '18:00'
        self.PeakEnd = '20:00'

        self.date = None
        self.last_updated_time = None

        # Initial data containers and setup
        self.price_history = None
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
        self.t_dis_start1 = datetime.strptime(
            self.DisChgStart1, '%H:%M').time()
        self.t_dis_end1 = datetime.strptime(
            self.DisChgEnd1, '%H:%M').time()
        self.t_peak_start = datetime.strptime(
            self.PeakStart, '%H:%M').time()
        self.t_peak_end = datetime.strptime(
            self.PeakEnd, '%H:%M').time()
        self.init_price_history()

    def init_price_history(self):
        self.price_history = [20 for i in range(self.LookBackBars)]

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

        # Buy and sell price based on historical data
        buy_price, sell_price = np.percentile(
            self.price_history, [self.BuyPct, self.SellPct])

        # peak_price = np.percentile(self.price_history, self.PeakPct)
        # use hard coded peak price for now
        peak_price = self.PeakPrice
        is_high_price = False
        if current_price > sell_price:
            current_feedin_price = current_price - self.price_gap
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

        if current_feedin_price > peak_price and current_pv < current_usage:
            command = {'command': 'Discharge',
                       'power': power, 'anti_backflow': False}
        return command, is_high_price

    def _is_charging_period(self, t):
        return t >= self.t_chg_start1 and t <= self.t_chg_end1

    def _is_discharging_period(self, t):
        # return True
        return (t >= self.t_dis_start2 and t <= self.t_dis_end2) or (t >= self.t_dis_start1 and t <= self.t_dis_end1)

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
        df_30min = df_day[['time', 'price', 'power_delta']
                          ].resample('30min', on='time').median()
        return df_30min.reset_index()

    def get_discharge_percentage(self):
        discharge_days = self.df[self.df['action']
                                 == 'Discharge']['time'].dt.date.nunique()
        total_days = self.df['time'].dt.date.nunique()
        discharge_percentage = (discharge_days / total_days) * 100
        return discharge_percentage


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    mock = MockData()
    file_names = mock.read_csv_from_dir('redx_data')
    st.set_page_config(page_title=None, page_icon=None, layout="wide",
                       initial_sidebar_state="auto", menu_items=None)
    st.title("Peak Valley Simulation Analysis")

    # Run the simulation
    with st.sidebar:
        on = st.toggle('Toggle Time Mode', help="Time Mode is a mode that uses a fixed time window for charging and discharging. When the time is within the window, the battery will charge or discharge based on the price. When the time is outside the window, the battery will be idle.")
        time_mode_start = st.time_input(
            "Time Mode Discharging Start", datetime.strptime('16:00', '%H:%M').time())
        time_mode_end = st.time_input(
            "Time Mode Discharging End", datetime.strptime('19:00', '%H:%M').time())
        selected_filename = st.selectbox("Select a filename", file_names)
        st.write("Simulation Parameters")
        buy_percentile = st.slider(
            "Buy percentile", 1, 100, 30, help="Start to charge battery when price is below this value")
        sell_percentile = st.slider(
            "Sell percentile", 1, 100, 30, help="Start to discharge battery when price is above this value")
        peak_percentile = st.slider(
            "Peak percentile", 1, 100, 70, help="Discharge w/o anti-backflow when price is above this value")
        peak_price = st.slider("Peak price",  1, 1000,
                               1000, help="Daytime peak price threshold")
        high_price_gap = st.slider("High Price gap",  1, 1000,
                                   10, help="The gap between the feedin price and the buy/sell price when the price is high (>sell price)")
        look_back_days = st.slider(
            "Look Back Days", 1, 20, 1, help="The buy/sell price is calculated based on the historical data in the past X days")
        st.write("Time window for charging and discharging")
        discharge_window_start = st.time_input(
            "Discharging Window Start", datetime.strptime('17:00', '%H:%M').time())
        discharge_window_end = st.time_input(
            "Discharging Window End", datetime.strptime('23:55', '%H:%M').time())
        charge_window_start = st.time_input(
            "Charging Window Start", datetime.strptime('08:00', '%H:%M').time())
        charge_window_end = st.time_input(
            "Charging Window End", datetime.strptime('16:00', '%H:%M').time())
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
                          time_mode_discharge_start=time_mode_start.strftime(
                              '%H:%M'),
                          time_mode_discharge_end=time_mode_end.strftime('%H:%M')
                          )
    ret = simulator.run_simulation()
    df = ret["df"]

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.sort_stats('time').print_stats(10)
    stats.sort_stats('cumulative').print_stats(10)

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
                '<div class="usage-box data-point">MONEY MADE</div>', unsafe_allow_html=True)

        df_discharge = df[df['action'] == 'Discharge']
        df_discharge = df_discharge[df_discharge['power_delta'] < 0]
        hist_discharge = np.histogram(df_discharge['price'], bins=[
            0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 1000], weights=0.1*df_discharge['power_delta'])
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
    with col2:
        st.subheader("Battery Discharging Distribution by Time")
        sim_visualizer = SimulationVisualizer(df)

        d = st.date_input("Please choose date", date(2025, 1, 17))
        df_30min = sim_visualizer.get_resampled_data(d)
        if not df_30min.empty:
            st.bar_chart(df_30min, x='time', y='power_delta')
            st.bar_chart(df_30min, x='time', y='price')
