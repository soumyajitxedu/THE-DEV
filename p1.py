#Part 1: Conditional Statements (if, elif, else)
#Ticket Pricing 🎟️: Write a program that asks the user for their age and prints the ticket price based on the following rules:

#Age under 5: Free

#Age 5 to 12: $10

#Age 13 to 65: $20

#Age over 65: $15

#Number Sign ➕➖: Write a program that takes a number as input and prints whether it's positive, negative, or zero.

#Leap Year Checker 📅: A year is a leap year if it is divisible by 4, unless it is divisible by 100 but not by 400. Write a program that asks for a year and determines if it's a leap year.
age = (input("input your age "))

while True:
    try:
        age = int(input("Enter your age: "))
        if age < 0:
            print("Age can't be negative.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter a valid age.")

if age < 5:
    print("Free")
elif age <= 12:
    print("$10")
elif age <= 65:
    print("$20")
else:
    print("$15")
