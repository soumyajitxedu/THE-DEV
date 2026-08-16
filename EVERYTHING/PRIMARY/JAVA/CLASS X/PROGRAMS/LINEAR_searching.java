import java.util.Scanner;

public class LINEAR_searching {
    @SuppressWarnings("ConvertToTryWithResources")
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        // Taking array size
        System.out.print("Enter size of array: ");
        int n = sc.nextInt();
        
        // Creating array
        int[] arr = new int[n];
        
        // Taking elements
        System.out.println("Enter " + n + " elements:");
        for (int i = 0; i < n; i++) {
            arr[i] = sc.nextInt();
        }
        
        // Taking target to search
        System.out.print("Enter number to search: ");
        int target = sc.nextInt();
        
        // Linear Search
        boolean found = false;
        int position = -1;  // -1 means not found
        
        for (int i = 0; i < n; i++) {
            if (arr[i] == target) {
                found = true;
                position = i;
                break;  // Stop searching after finding first occurrence
            }
        }
        
        // Display result
        if (found) {
            System.out.println(target + " found at index: " + position);
        } else {
            System.out.println(target + " not found in the array");
        }
        
        sc.close();
    }
}