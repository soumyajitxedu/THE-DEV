import java.util.Scanner;
class CloudStorage
{
    int acno;
    int space;
    double bill;
    Scanner scan = new Scanner(System.in);
    void accept(){
        System.out.println("ENTER THE ACCOUNT NUMBER");
        acno = scan.nextInt();
        System.out.println("Enter Storage ");
        space = scan.nextInt();



    }
    void calculate(){
        if(space <= 15)
        {
            bill = space * 15;

        }
        else 
            if(space > 15 && space <= 30)
                bill = 15*15+(space-15)*13;
            else
                bill = 15 * 15 + 13 + (space - 30)*11;

    }
    void Display(){
        System.out.println("ACCOUNT NUMBER IS : " + acno);
        System.out.println("Storage number : " + space);
        System.out.println("The bill is : " + bill);

    }
    void main() {
        CloudStorage c = new CloudStorage();
        c.accept();
        c.calculate();
        c.Display();

    }
}