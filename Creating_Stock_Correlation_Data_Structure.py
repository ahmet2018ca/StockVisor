'''
https://finance.yahoo.com/chart/AAPL
https://finance.yahoo.com/chart/GOOGL
https://finance.yahoo.com/chart/NVDA
'''
import csv, glob, numpy


#GLOBAL VARIABLES

previous_closing_global = None


#Change % of current day's open to current day's close 
def change_percentage_of_day(open_value, close_value):

    percentage_change = ((close_value - open_value) / open_value) * 100
    
    return round(percentage_change, 2) #rounds to two decimal points
    


def change_of_day(open_value, close_value):

    change_of_day = float(close_value) - float(open_value)
    change_of_day = round(change_of_day, 2)
    
    return change_of_day


def previous_day_close(previous_closing_local):

    global previous_closing_global


    if previous_closing_global == None:
        previous_closing_global = previous_closing_local

        return "NA"
    
    else:
        x = previous_closing_global
        previous_closing_global = previous_closing_local
        return x


#Function that changes (str) "57450710" ---> "57.45M" (str)
def format_volume(number_str):
    
    value = int(number_str)
    str_value =  f"{value/10**6:.2f}M" # A bit of math involved :D
    int_value = int(number_str)
    volume_tuple = (int_value, str_value)
    
    return volume_tuple
  
#Function that changes (str) "147.8902" ----> 147.89 (float)
def parse_float(value):
    
    value = float(value)
    rounded_value = round(value, 2)  
    
    return rounded_value


#Finds NASDAQ stock files
def find_stock_data():
    
    nasdaq_files = glob.glob("NASDAQ_STOCK*.csv")

    return nasdaq_files



def create_stocks_correlation_data_structure(stock_files):

    stocks_correlation_data_dict = {}
  
    for file in stock_files:
        
        stock_name = file.split("_")[2]    #NASDAQ_STOCK_APPL_DATA

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
                
    
                if stock_name not in stocks_correlation_data_dict:
                    stocks_correlation_data_dict[stock_name] = {}
                
                if date not in stocks_correlation_data_dict[stock_name]:
                    stocks_correlation_data_dict[stock_name][date] = {}
                
                stocks_correlation_data_dict[stock_name][date]["Close"] = closing_price
                stocks_correlation_data_dict[stock_name][date]["Open"] = opening_price
                stocks_correlation_data_dict[stock_name][date]["Previous Close"] = previous_close_price
                stocks_correlation_data_dict[stock_name][date]["Volume"] = volume
                stocks_correlation_data_dict[stock_name][date]["Change % Of Day"] = change_percentage_of_open_close
                stocks_correlation_data_dict[stock_name][date]["Change Of Day"] = change_of_open_close

                

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

        matrices = numpy.corrcoef(x, y)
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
    print_stocks_info(stock_correlation_data_dict)  # This is to print all stock data before the math of correlation coefficent finder
    #correlation_coefficent_finder(stock_correlation_data_dict)

    
    return print_stocks_info

main()





