
//### Class: `CloudStorage`

//Define a class named `CloudStorage` with the following specifications:

//#### **Member Variables**
//* **`int acno`**: Stores the user's account number.
//* **`int space`**: Stores the amount of storage space in GB purchased by the user.
//* **`double bill`**: Stores the total price to be paid by the user.

//#### **Member Methods**
//* **`void accept()`**: Prompts the user to input their account number and storage space using Scanner class methods only.
//* **`void calculate()`**: Calculates the bill total price based on the storage space purchased using the following pricing table:

//| Storage Range | Price per GB (Rs) |
//| :--- | :--- |
//| First 15 GB | 15 |
//| Next 15 GB | 13 |
//| Above 30 GB | 11 |

//* **`void display()`**: Displays the account number, storage space, and bill to be paid.

//#### **Execution**
//* In the `main` method, create an object of the `CloudStorage` class and invoke the `accept()`, `calculate()`, and `display()` methods in sequence to demonstrate the functionality of the class.
import java.util.Scanner;
public class CloudStorage {
    int acno;
    int space;
    double bill;
    void accept(){
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter account number and space in GB");
        acno = sc.nextInt();
        System.out.println("Enter space in GB");
        space = sc.nextInt();
        System.out.println("Account number: " + acno);
        System.out.println("Storage space: " + space + " GB");
        

    }

    void calculate(){
        if (space <= 15){
            System.out.println("Price per GB: 15 Rs");
        }
        else if (space <=30)
        {
            System.out.println("price per GB: 13 Rs");


        }
        else {
            System.out.println("Price per GB: 11 Rs");
            bill = 15 * 15 + 15 * 13 + (space - 30) * 11;
            System.out.println("total bill : " + bill + " Rs");

        }

    }
    void display(){
        System.out.println("account number : "+ acno);
        System.out.println("storage space :"+ space + " GB");
        System.out.println("Total bill: Rs " + bill);

    }

    public void main(String[] args){
        CloudStorage obj = new CloudStorage();
        obj.accept();
        obj.calculate();
        obj.display();

    }

}