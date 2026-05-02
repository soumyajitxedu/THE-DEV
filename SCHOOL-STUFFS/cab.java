import java.util.Scanner;

public class CloudStorage {
    // Member Variables
    int acno;    // stores the user's account number
    int space;   // stores the amount of storage space in GB
    double bill; // stores the total price to be paid

    // Method to accept user input
    void accept() {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter Account Number: ");
        acno = sc.nextInt();
        System.out.print("Enter storage space in GB: ");
        space = sc.nextInt();
    }

    // Method to calculate bill based on tiers
    void calculate() {
        // Pricing Table:
        // First 15 GB: 15 Rs/GB
        // Next 15 GB: 13 Rs/GB
        // Above 30 GB: 11 Rs/GB
        
        if (space <= 15) {
            bill = space * 15;
        } else if (space <= 30) {
            bill = (15 * 15) + ((space - 15) * 13);
        } else {
            bill = (15 * 15) + (15 * 13) + ((space - 30) * 11);
        }
    }

    // Method to display account details
    void display() {
        System.out.println("Account Number: " + acno);
        System.out.println("Space Purchased: " + space + " GB");
        System.out.println("Total Bill: Rs " + bill);
    }

    // Main method to create an object and invoke methods
    public static void main(String[] args) {
        CloudStorage obj = new CloudStorage();
        obj.accept();
        obj.calculate();
        obj.display();
    }
}