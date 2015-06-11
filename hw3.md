##Homework 3##

```
#1. Read chipotle data and store it in a list of lists
import csv
with open('chipotle.tsv', 'rU') as f:
    data = [row for row in csv.reader(f, delimiter = '\t')]

#2. Separate the header and data
header = data[0]
data = data[1:] 
# len(data) does not equal to 4623 after running commands above

#3. Calculate average price of an order
#each order can be one or more rows.  A row can have quantity
#greater than 1.  Quantity does not figure into item_price.
#pseudo: A. item_price: remove '$' and convert to float
#        B. sum item_price
#        C. sum(item_price) divide by max(order_id) and make sure there is 
#           no missing order_id
order_count = len(set([row[0] for row in data])) 
item_prices = [float(row[4][1:]) for row in data]
avg_order_price = round(sum(item_prices)/order_count,2)
print "Average order price is $ %f" %(avg_order_price)
#$18.81 is average order price.

#4. Create a list of all unique sodas and soft drinks
#pseudo: A. filter for item_name == "Canned Soda" or "Canned Soft Drink"
#        B. find all unique choice_description
canned_drinks = []
for row in data:
    if row[2] == 'Canned Soda':
        canned_drinks.append(row[3])
    elif row[2] == 'Canned Soft Drink':
        canned_drinks.append(row[3])
print set(canned_drinks)
#Lemonade, Dr. Pepper, Diet Coke, Nestea, Mountain Dew, Diet
#Dr. Pepper, Coke, Coca Cola, Sprite

#5. Calculate the average number of toppings per burrito
#pseudo: A. filter for item_name last 7 characters are 'burrito'
#        B. count number of toppings for each row from A 
#        C. divide B. by length of A
burritos = []
for row in data:
    if row[2][-7:] == "Burrito":
        burritos.append(row[3])
burritos = [row.split(',') for row in burritos] #convert toppings, a string, to
#a list
round(sum([len(row) for row in burritos])/float(len(burritos)), 2)
#average topping per burrito is 5.4
 
#6. Create a dictionary.  Keys represent chip orders and values represent total
#number of orders.
from collections import defaultdict
chips = []
#create a list of chips and order quantities
for row in data:
    if 'Chips' in row[2]:
        chips.append([row[2],row[1]])
    elif 'chips' in row[2]: 
        chips.append([row[2],row[1]])
#use defaultdict to create key (chips) and value (quantities converted to int)
d = defaultdict(list)
for c, q in chips:
    d[c].append(int(q))
#print sum of order quantity for each key
print {c:sum(q) for c,q in d.items()}
#{0: 0, 1: 0, 'chips side': 0, 'Chips and Roasted Chili-Corn Salsa': 18,
# 'Chips and Mild Fresh Tomato Salsa': 1, 'Chips and Tomatillo-Red Chili Salsa'
#: 25, 'Chips and Guacamole': 506, 'Chips and Fresh Tomato Salsa': 130, 
#'Side of Chips': 110, 'Chips and Tomatillo-Green Chili Salsa': 33, 
#'Chips and Tomatillo Red Chili Salsa': 50, 
#'Chips and Roasted Chili Corn Salsa': 23, 'Chips': 230, 
#'Chips and Tomatillo Green Chili Salsa': 45}

```