##Homework: Command Line Introduction##
1. chitpotle.tsv analysis
	i. What does each column mean? each column appears to be data attributes of a record 
	(one or one of several order items that make up an order). There are five attributes: order_id, 
	quantity, item_name, choice_description, and item_price.  item_price appears to be a subtotal price (i.e.,
	unit price, which is not provided, multiplied by quantity, which is provided).  This conclusion is based on 
	visual inspection of records with the identical item_name and choice_description that have different prices. 
  
	What does each row mean? a record (discrete item that make up the whole or a part of an order).

	```
	head chipotle.tsv
	tail chipotle.tsv
	grep -i 'chicken bowl' chipotle.tsv | head 
	```

	ii. 1834 orders
	```
	tail chipotle.tsv
	```
	iii. 4623 lines
	```
	wc -l chipotle.tsv
	```
	iv. chicken burritos were ordered 553 times vs. steak burrito = 368 times
	```
	grep -i 'steak burrito' chipotle.csv | wc -l
	grep -i 'chicken burrito' chipotle.csv | wc -l 
	```
2. find files
	
	./data/airlines.csv
	./data/chipotle.tsv
	./data/sms.tsv
	```
	find . -name *.?sv
	```
3. count the number of occurances of 'dictionary' (case insensitive)
	couldn't get this
	```
	grep -ri 'dictionary' . 	
	##what command do I use to count the number of 'dictionary'?
	```
