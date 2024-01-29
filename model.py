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

    def __init__(self, date_start='2022-01-01', date_end='2022-01-03'):
        self.date_start = date_start
        self.date_end = date_end
        self.df = pd.read_csv('friend.csv')
        self._prepare_data()

    def _prepare_data(self):
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df[(self.df['time'] >= self.date_start) & (
            self.df['time'] < self.date_end)]

    def generate_solar_data(self):
        x = np.linspace(0, 288, 289)
        mean = 144
        std = 50
        y = np.exp(-((x - mean) ** 2) / (std ** 2))*4000
        return y

    def get_data_generator(self):
        for _, row in self.df.iterrows():
            yield {
                'current_price': row['price'],
                'current_datetime': row['time'],
                'current_pv': self.generate_solar_data()[row['time'].hour * 12 + row['time'].minute // 5],
                'current_usage': row['usage']
            }


class Simulator:

    def __init__(self, date_start='2022-01-01', date_end='2022-01-31', **kwargs):
        self.model = PeakValleyScheduler(**kwargs)
        self.battery_stats = Battery(max_capacity=5000)
        self.mock_data = MockData(date_start, date_end)
        self.cost_wo_battery = []
        self.cost_w_battery = []

    def run_simulation(self):
        log_data = []
        df = pd.DataFrame(
            columns=['time', 'battery_soc', 'price', 'action', 'cost', 'cost_savings'])
        max_power_feedin = []
        for mock_data in self.mock_data.get_data_generator():
            # 1. Get current price and other inputs
            price = mock_data['current_price']
            now = mock_data['current_datetime'].strftime('%H:%M')
            pv = mock_data['current_pv']/1000
            usage = mock_data['current_usage']
            # 2. Model step
            command = self.model.step(
                price, now, usage, self.battery_stats.state_of_charge, pv, device_type="2505")
            # 3. Update battery state
            power_delta = self.battery_stats.get_actual_power_delta(
                command, usage)['power_delta']
            if command.get('anti_backflow', False) == False and command.get('command', 'Idle') == 'Discharge':
                max_power_feedin.append(1)
            elif command.get('anti_backflow', False) == True and command.get('command', 'Idle') == 'Discharge':
                max_power_feedin.append(0)
            action = command.get('command', 'Idle')

            # 4. Update cost
            price_dollar = price / 100
            self.cost_wo_battery.append(price_dollar * usage)
            net_usage = usage + power_delta / 1000
            
            if net_usage < 0:
                cost = abs(price_dollar - 0.1)*net_usage # feedin price is 10 cents lower than the price 
            else:
                cost = price_dollar * net_usage
            self.cost_w_battery.append(cost)

            # Log the action
            log_data.append([mock_data['current_datetime'], self.battery_stats.state_of_charge,
                            price, action, self.get_power_cost(), self.get_power_cost_savings(), power_delta])
        # Print the log data in a table format
        df = pd.DataFrame(log_data, columns=[
                          'time', 'battery_soc', 'price', 'action', 'cost', 'cost_savings', 'power_delta'])
        ret = {"df": df, "total_saved": self.get_power_cost_savings() / self.get_power_cost()*100,
               "total_backflow": sum([1 for i in max_power_feedin if i])/len(max_power_feedin)*100}
        return ret
        # headers = ["Time", "Battery Status", "Price",
        #            "Action", "Cost", "Cost Savings\n(Dollars)"]
        # return tabulate(log_data, headers=headers, tablefmt="fancy_grid"), tabulate([[f"{sum([1 for i in max_power_feedin if i])/len(max_power_feedin)*100:.2f}%", f"{self.get_power_cost_savings() / self.get_power_cost()*100:.2f}%"]],
        #   headers=["Percentage of time with anti-backflow", "Overall percentage of cost savings"], tablefmt="fancy_grid"), df

    def get_power_cost(self):
        return sum(self.cost_wo_battery)

    def get_power_cost_savings(self):
        cost_original = sum(self.cost_wo_battery)
        cost_savings = sum(self.cost_w_battery) - cost_original
        return cost_savings


class PeakValleyScheduler():
    def __init__(self, buy_percentile=30, sell_percentile=65, peak_percentile=90, peak_price=200, look_back_days=2, jc_param1=30, jc_param2=50, jc_param3=30):
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
        self.LookBackBars = look_back_days * 48
        self.ChgStart1 = '04:00'
        self.ChgEnd1 = '16:00'
        self.DisChgStart2 = '16:05'
        self.DisChgEnd2 = '23:55'
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
        current_feedin_price = current_price - 10
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

        command = {"command": "Idle"}

        if self._is_charging_period(current_time) and (current_price <= buy_price or current_pv > current_usage):
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
        return command

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
        df_day = self.df[(self.df['time'] >= start_date) & (self.df['time'] < end_date)]
        df_30min = df_day[['time','price', 'power_delta']].resample('30min', on='time').median()
        return df_30min.reset_index()

    def get_discharge_percentage(self):
        discharge_days = self.df[self.df['action'] == 'Discharge']['time'].dt.date.nunique()
        total_days = self.df['time'].dt.date.nunique()
        discharge_percentage = (discharge_days / total_days) * 100
        return discharge_percentage
    

if __name__ == '__main__':
    st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
    st.title("Peak Valley Simulation Showcase")
    st.write("This is a showcase page for the Peak Valley Simulation.")
    

    # Run the simulation
    with st.sidebar:
        st.write("Simulation Parameters")
        buy_percentile = st.slider(
            "Buy percentile", 1, 100, 30, help="Start to charge battery when price is below this value")
        sell_percentile = st.slider(
            "Sell percentile", 1, 100, 30, help="Start to discharge battery when price is above this value")
        peak_percentile = st.slider(
            "Peak percentile", 1, 100, 30, help="Discharge w/o anti-backflow when price is above this value")
        peak_price = st.slider("Peak price",  1, 1000,
                               200, help="Daytime peak price threshold")
        look_back_days = st.slider(
            "Look Back Days", 1, 20, 2, help="The buy/sell price is calculated based on the historical data in the past X days")
        jc_param1 = st.slider(
            "Weak Discharging Threshold", 1, 1000, 30, help="The value of fixed weak discharging (anti-backflow enabled) price threshold")
        jc_param2 = st.slider(
            "Max Discharging Threshold", 1, 1000, 50, help="The value of fixed max discharging (anti-backflow disabled) price threshold")
        jc_param3 = st.slider(
            "Charging Price Threshold", 1, 1000, 30, help="The value of fixed charging price threshold")

    simulator = Simulator(date_start='2021-07-01', date_end='2022-03-03', buy_percentile=buy_percentile,
                          sell_percentile=sell_percentile, peak_percentile=peak_percentile, peak_price=peak_price, look_back_days=look_back_days, jc_param1=jc_param1, jc_param2=jc_param2, jc_param3=jc_param3)
    ret = simulator.run_simulation()
    df = ret["df"]

    col1, col2 = st.columns(2)
    # Display command line output using tabulate
    with col1:
        st.subheader("Simulation Results")
        st.write("The following table shows the simulation results on a 30-min freq from the start date to the end date.")
        st.write("time: the time of the day battery soc: the state of charge of the battery price: the price of the electricity action: the action taken by the battery cost: the cost of electricity without battery cost_savings: the cost of electricity with battery")
        st.write(df)
        st.subheader(f"Total cost savings:")
        text = f"${ret['total_saved']:.2f}"
        st.markdown(f"<h3 style='text-align: center; color: #0068c9;'>{text}</h1>",
                    unsafe_allow_html=True)
        st.write(
            "this shows the percentage of cost savings with the battery against w/o battery.")
        st.subheader(
            f"Percentage of time with anti-backflow disabled:")
        text = f"{ret['total_backflow']:.2f}%"
        st.markdown(f"<h3 style='text-align: center; color: #0068c9;'>{text}</h1>",
                    unsafe_allow_html=True)
        st.write("this shows the percentage of time when the battery is discharging without anti-backflow enabled. This is a good indicator of how much the battery is trying to make money from high price rather than cover the user usage.")

        st.subheader(
            f"Battery Discharging Percentage:")
        percentage = SimulationVisualizer(df).get_discharge_percentage()
        text = f"{percentage:.2f}%"
        st.markdown(f"<h3 style='text-align: center; color: #0068c9;'>{text}</h1>",
                    unsafe_allow_html=True)
        df_discharge = df[df['action'] == 'Discharge']
        df_discharge = df_discharge[df_discharge['power_delta'] < 0]
        hist_discharge = np.histogram(df_discharge['price'], bins=[
                                    0, 10, 20, 30, 40, 50, 60, 70, 80, 300], weights=0.1*df_discharge['power_delta'])
        df_hist = pd.DataFrame(hist_discharge)
        df_hist_inverted = df_hist.T
        df_hist_inverted.columns = ['y', 'x']
        df_hist_inverted['y'] = -df_hist_inverted['y']
        st.markdown("---")
        st.subheader("Discharge price distribution")
        st.write("The following chart shows the distribution of the feedin price when the battery is discharging.")
        st.bar_chart(df_hist_inverted, x='x', y='y')
    with col2:
        st.subheader("Battery Discharging Distribution by Time")
        sim_visualizer = SimulationVisualizer(df)

        d = st.date_input("Please choose date", date(2021, 10, 1))
        df_30min = sim_visualizer.get_resampled_data(d)
        st.bar_chart(df_30min, x='time', y='power_delta')
        st.bar_chart(df_30min, x='time', y='price')
