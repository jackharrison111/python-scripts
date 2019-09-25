
"""

Jack Harrison - 16/05/2019 16:50

A simple code to perform a bubble sort on an array of integers.


I have chosen the simplest sorting algorithm, Bubble sort.
Bubble sort works by counting along the array, comparing each position with its adjacent
value and swapping the two over if desired order is wrong. The algorithm then repeats this 
checking until the array is fully sorted.


Bubble sort is a very ineffient algorithm for sorting, yet very easy to implement. 
At worst case, the whole array is in reverse order and the efficiency of the algorithm is of 
order O(n^2) (requiring n swaps n times). Bubble sort also has an average efficiency of order O(n^2)
and hence is not a good algorithm for use on large datasets. In the best case scenario, the 
array is already sorted and the algorithm requires n checks, leaving it with order O(n).

A better algorithm to use is the Quicksort algorithm. This involves taking pivot positions and 
sorting the array around the pivot value, and then repeatedly taking new pivot positions. The
choice of pivot positions also depends on the type of Quicksort and can affect the efficiency.
Quicksort has an average O(n log n), and at worst case makes O(n^2) - equal to the Bubble sort.


"""

#bubble sort function
def Sort(values = [], *args):
    #repeat swaps until complete
    repeat = True
    while repeat == True:
     swaps = 0
     #for each position in the array
     for i in range(0,len(values)-1):
      #swap if the next value is smaller
      if values[i+1] < values[i]:
       values[i+1],values[i] = values [i], values[i+1]
       swaps += 1
      else:
       continue
     #if no swaps are needed then sort is complete
     if swaps == 0:
         repeat = False
    
    return values

#example array
values = [0,5,4,3,73,5,7,4,2,5,6,120,-1]

#sort the values
sortedValues = Sort(values)

#show the sort works
print(sortedValues)