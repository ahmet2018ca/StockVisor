'''
https://finance.yahoo.com/chart/AAPL
https://finance.yahoo.com/chart/GOOGL
https://finance.yahoo.com/chart/NVDA
'''

import csv, glob, mplcursors, matplotlib
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


#Change % of current day's open to current day's close 
def change_percentage_of_day(open_value, close_value):
    percentage_change = ((close_value - open_value) / open_value) * 100
    return round(percentage_change, 2) #rounds to two decimal points
        
def change_of_day(open_value, close_value):
    change_of_day = float(close_value) - float(open_value)
    change_of_day = round(change_of_day, 2)
    
    return change_of_day

#GLOBAL VARIABLE
previous_closing_global = None
def previous_day_close(previous_closing_local):
    global previous_closing_global
    if previous_closing_global == None:
        previous_closing_global = previous_closing_local
        return None
    else:
        x = previous_closing_global
        previous_closing_global = previous_closing_local
        return x

def days_range(open_price, close_price):
    return f"{open_price} - {close_price}"
    
def calculate_three_month_average_volume(stocks_dict):
    three_month_averages = {}
    
    for stock in stocks_dict:
        print(stock)
        # Sort the dates to find the latest date
        dates = sorted(stocks_dict[stock].keys(), key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
        latest_date = dates[-1]  # Latest date
        latest_date_datetime = datetime.strptime(latest_date, '%Y-%m-%d')
        # Calculate the start date of the 3 months interval
        three_months_ago = latest_date_datetime - timedelta(days=90)
        # Filter dates that are within the last 3 months
        recent_volumes = [stocks_dict[stock][date]['Volume'][0] for date in dates if datetime.strptime(date, '%Y-%m-%d') >= three_months_ago]  
        # Calculate average if there are any volumes collected
        if recent_volumes:
            average_volume = np.mean(recent_volumes)
            average_volume = int(average_volume)
        else:
            average_volume = 0   
        # Store the average in the dictionary
        three_month_averages[stock] = average_volume
    return three_month_averages

#Function that changes (str) "57450710" ---> "57.45M" (str)
def format_volume(number_str):
    
    value = int(number_str)
    str_value =  f"{value/10**6:.2f}M" # A bit of math involved :D
    int_value = int(number_str)
    volume_tuple = (int_value, str_value)    
    return volume_tuple


def fifty_two_week_range(stocks_data_dict):
    # Get 52 weeks ago date
    for stock_name in stocks_data_dict:
        for date in stocks_data_dict[stock_name]:
            closing_price = stocks_data_dict[stock_name][date]["Close"]
            one_year = datetime.strptime(date, '%Y-%m-%d') - relativedelta(years=1)
            fifty_two_weeks_ago = one_year - relativedelta(weeks=52)
            for i in range(8):  
                if fifty_two_weeks_ago.strftime('%Y-%m-%d') in stocks_data_dict[stock_name]:
                    fifty_two_week_close = stocks_data_dict[stock_name][fifty_two_weeks_ago.strftime('%Y-%m-%d')]["Close"]
                    fifty_two_week_range = f"{fifty_two_week_close} - {closing_price}"
                    stocks_data_dict[stock_name][date]["52 Week Range"] = fifty_two_week_range
                    break
                elif i == 7:
                    fifty_two_week_range = "N/A (data unavailable)"
                    stocks_data_dict[stock_name][date]["52 Week Range"] = fifty_two_week_range 
                else:
                    fifty_two_weeks_ago += timedelta(days=1)
    return stocks_data_dict

def one_week_range(stocks_data_dict):
    for stock_name in stocks_data_dict:
        for date in stocks_data_dict[stock_name]:
            current_date = datetime.strptime(date, '%Y-%m-%d')
            one_week_ago = current_date - timedelta(days=7)

            # Try to find the closest date within the past 7 days with available data
            for i in range(7):  # Search up to 7 days back
                one_week_ago_str = (one_week_ago - timedelta(days=i)).strftime('%Y-%m-%d')
                if one_week_ago_str in stocks_data_dict[stock_name]:
                    one_week_ago_close = stocks_data_dict[stock_name][one_week_ago_str]["Close"]
                    current_close = stocks_data_dict[stock_name][date]["Close"]
                    one_week_range_str = f"{one_week_ago_close} - {current_close}"
                    stocks_data_dict[stock_name][date]["1 Week Range"] = one_week_range_str
                    break
            else:
                # If no data is available within the past week
                stocks_data_dict[stock_name][date]["1 Week Range"] = "N/A (data unavailable)"
    
    return stocks_data_dict

#Function that changes (str) "147.8902" ----> 147.89 (float)
def parse_float(value):  
    return round(float(value), 2)

#Finds NASDAQ stock files
def find_stock_data():
    return glob.glob("NASDAQ_STOCK*.csv")



def create_stocks_data_structure(stock_files):
    stocks_data_dict = {}
    for file in stock_files: 
        stock_name = file.split("_")[2][:-4]    #NASDAQ_STOCK_APPL.csv --> APPL
        with open(file) as file_in:
            reader = csv.DictReader(file_in)           
            for line in reader:
                date = line["Date"]  
                volume = format_volume(line["Volume"])
                closing_price = parse_float(line["Close"])
                opening_price = parse_float(line["Open"])
                previous_close_price = previous_day_close(closing_price)
                change_percentage_of_open_close = change_percentage_of_day(opening_price, closing_price)
                change_of_open_close = change_of_day(opening_price, closing_price)
                todays_range = days_range(opening_price, closing_price)
                # Add data to dictionary
                if stock_name not in stocks_data_dict:
                    stocks_data_dict[stock_name] = {}
                stocks_data_dict[stock_name][date] = {
                    "Close": closing_price,
                    "Open": opening_price,
                    "Previous Close": previous_close_price,
                    "Volume": volume,
                    "Change % Of Day": change_percentage_of_open_close,
                    "Change Of Day": change_of_open_close,
                    "Day's Range": todays_range,
                }
    return stocks_data_dict

def print_stocks_info(stocks_dict):    
    for stocks in stocks_dict:
        ("\n"*3)
        print("*"*125)
        ("\n")
        print(stocks)
        print("\n") 

        for dates in stocks_dict[stocks]:
            print(dates)
            print(stocks_dict[stocks][dates])
            print("\n") 

def correlation_coefficent_finder(stocks_dict): #will need to add number of days once I let the user pick in which dates time
    for stocks in stocks_dict:
        x = []
        y = []
        for dates in stocks_dict[stocks]:
            x.append(stocks_dict[stocks][dates]["Volume"])
            y.append(stocks_dict[stocks][dates]["Change Of Day"])
        matrices = np.corrcoef(x, y)
        r = matrices[0][1]
        stock = stocks[:-4]
        print(f"Stock is {stock} || correlation coefficent between Volume and Change % Day is  {round(r,4)}")
    
    
    '''
        if len(x) != len(y):
            return None  # Return None if lists have different lengths
        N = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
        sum_x2 = sum(x_i**2 for x_i in x)
        sum_y2 = sum(y_i**2 for y_i in y)
        
        numerator = (N * sum_xy) - (sum_x * sum_y)
        denominator = ((N * sum_x2 - sum_x**2) * (N * sum_y2 - sum_y**2))**0.5
        
        if denominator == 0:
            return None  # Return None if division by zero occurs
        
        r = numerator / denominator

        LONGER WAY OF DOING CORRELATION COEFFICENT
    '''


def update_stock_dict(stocks_dict):
    updated_dict = fifty_two_week_range(stocks_dict)
    updated_dict = one_week_range(updated_dict)
    return updated_dict

def plot_interactive_line_graph_with_details(df, stock_name):
    # Ensure the date column is a datetime type and data is sorted
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)  # Reset index to ensure integer indexing
    # Setting up the plot
    fig, ax = plt.subplots()
    line, = ax.plot(df['Date'], df['Close'], label=f'Closing Prices for {stock_name}')
    ax.set_title(f'Interactive Line Graph for {stock_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.legend()
    # Prepare an annotation for interactivity
    annot = ax.annotate("", xy=(0.5, 0.5), xycoords='axes fraction', xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=None)  # Remove arrow
    annot.set_visible(False)

    # Function to update the annotation
    def update_annot(ind):
        # Get the index of the point hovered
        x, y = line.get_data()
        index = ind["ind"][0]
        details = df.iloc[index]
        text_lines = [
            f"Date: {details['Date'].strftime('%Y-%m-%d')}",
            f"Open: {details['Open']}",
            f"Close: {details['Close']}",
            f"Volume: {details['Volume'][1]}",
            f"Previous Close: {details.get('Previous Close', 'N/A')}",
            f"Day's Range: {details.get("Day's Range", 'N/A')}",
            f"1 Week Range: {details.get("1 Week Range", 'N/A')}",
            f"52 Week Range: {details.get('52 Week Range', 'N/A')}"
        ]
        text = "\n".join(text_lines)
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    # Hover event
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = line.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


# Assume df is your DataFrame loaded from your stock data
# For a specific stock, prepare DataFrame from your dict
# Example usage
# df = pd.DataFrame(stocks_data_dict['AAPL']).T
# df['Date'] = df.index
# plot_interactive_line_graph_with_details(df, 'AAPL')




def main(): 
    data_files = find_stock_data()
    stocks_data_dict = create_stocks_data_structure(data_files)
    
    #GET SPECIFIC VALUES
    #three_month_avg_volumes = calculate_three_month_average_volume(stock_correlation_data_dict)
    #correlation_coefficent_finder(stock_correlation_data_dict)
    updated_stocks_data_dict = update_stock_dict(stocks_data_dict)
    #print_stocks_info(updated_stocks_data_dict)  # This is to print all stock data before the math of correlation coefficent finder

    # Plotting data for each stock or a specific stock
    for stock_name, data in updated_stocks_data_dict.items():
        df = pd.DataFrame(data).T  # Convert data dictionary to DataFrame
        df['Date'] = df.index  # Set the index as date column
        plot_interactive_line_graph_with_details(df, stock_name)
        # You can break here if you only want to plot one stock
        # break
main()







