'''
https://finance.yahoo.com/chart/AAPL
https://finance.yahoo.com/chart/GOOGL
https://finance.yahoo.com/chart/NVDA
'''

#Question to ask yourself, do you want to just show the recent average? or the overall average?
#If overall


import csv, glob
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

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
    

#CHATGPT MAGIC :DDD
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
  
#Function that changes (str) "147.8902" ----> 147.89 (float)
def parse_float(value):
    
    return round(float(value), 2)

#Finds NASDAQ stock files
def find_stock_data():
    
    return glob.glob("NASDAQ_STOCK*.csv")


def create_stocks_correlation_data_structure(stock_files):

    stocks_correlation_data_dict = {}
  
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
                if stock_name not in stocks_correlation_data_dict:
                    stocks_correlation_data_dict[stock_name] = {}

                stocks_correlation_data_dict[stock_name][date] = {
                    "Close": closing_price,
                    "Open": opening_price,
                    "Previous Close": previous_close_price,
                    "Volume": volume,
                    "Change % Of Day": change_percentage_of_open_close,
                    "Change Of Day": change_of_open_close,
                    "Day's Range": todays_range,
                }

                # Get 52 weeks ago date
                one_year_ago = datetime.strptime(date, '%Y-%m-%d') - relativedelta(years=1)
                fifty_two_weeks_ago = one_year_ago - relativedelta(weeks=52)
                
                for i in range(10):  
                    if fifty_two_weeks_ago.strftime('%Y-%m-%d') in stocks_correlation_data_dict[stock_name]:
                        fifty_two_week_close = stocks_correlation_data_dict[stock_name][fifty_two_weeks_ago.strftime('%Y-%m-%d')]["Close"]
                        fifty_two_week_range = f"{fifty_two_week_close} - {closing_price}"
                        stocks_correlation_data_dict[stock_name][date]["52 Week Range"] = fifty_two_week_range
                    
                    elif i == 9:
                        fifty_two_week_range = "N/A (data unavailable)"
                        stocks_correlation_data_dict[stock_name][date]["52 Week Range"] = fifty_two_week_range
                    
                    else:
                        fifty_two_weeks_ago += timedelta(days=1)

                        
                
                


    return stocks_correlation_data_dict


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

        

def main():

    data_files = find_stock_data()
    stock_correlation_data_dict = create_stocks_correlation_data_structure(data_files)
    #print_stocks_info(stock_correlation_data_dict)  # This is to print all stock data before the math of correlation coefficent finder
    #correlation_coefficent_finder(stock_correlation_data_dict)
    three_month_avg_volumes = calculate_three_month_average_volume(stock_correlation_data_dict)
    print(three_month_avg_volumes)
    
    #return print_stocks_info

main()





