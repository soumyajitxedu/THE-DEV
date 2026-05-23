public class pattern
{
    public static void main(String[] args)
    {
        int rows = 5;
        for (int i = 1; i <= rows; i++){
            for (int j = i; j <= rows; j++){
                System.out.print(" ");
            }
            
            for (int k = 1;k  <= (2 * i -1); k++){
                System.out.print("*");

            }

            System.out.println(); // Print a new line after each row
        }
    }
}